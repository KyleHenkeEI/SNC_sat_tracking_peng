"""
Poisson multi-Bernoulli mixture (PMBM)-style tracker for passive video.

This module implements a **practical multi-hypothesis** extension of
:class:`TextbookPoissonMultiBernoulliTracker`: each frame, several **global
association patterns** (variants around the Hungarian solution) are retained
with **log-weights**, pruned to a fixed cap, and **MAP** components drive
overlays, ``tracks.txt``, and ``summary.json`` metrics.

This is **not** a full labeled-δ-PMBM reference implementation (no explicit
Poisson birth field over state space, no Murty k-best on the full assignment
polytope). It is intended as a **closer engineering approximation** to PMBM
than single-posterior Textbook PMB for RFI and experimentation.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from tracking_core.pmb import (
    BernoulliComponent,
    TextbookPoissonMultiBernoulliTracker,
    _clone_bernoulli_like,
    _mahalanobis_sq_position,
)


class PoissonMultiBernoulliMixtureTracker(TextbookPoissonMultiBernoulliTracker):
    """
    Textbook PMB measurement and Bernoulli update model, with a **mixture** over
    association hypotheses each frame.

    Parameters
    ----------
    pmbm_k_best:
        Maximum number of association variants to branch into **per parent**
        hypothesis (best Hungarian + forced-miss perturbations).
    pmbm_max_hypotheses:
        After branching and merging, keep at most this many mixture components
        (highest log-weight).
    """

    def __init__(
        self,
        input_video_path,
        output_video_path,
        *,
        pmbm_k_best: int = 5,
        pmbm_max_hypotheses: int = 10,
        **kwargs,
    ):
        self.pmbm_k_best = max(1, int(pmbm_k_best))
        self.pmbm_max_hypotheses = max(1, int(pmbm_max_hypotheses))
        super().__init__(input_video_path, output_video_path, **kwargs)
        # Log-weights (natural log); parallel list of Bernoulli component lists.
        self._pmbm_log_weights = np.array([0.0], dtype=np.float64)
        self._pmbm_hypothesis_components: list[list[BernoulliComponent]] = [[]]

    # --- mixture predict / prune -------------------------------------------------

    def predict_components(self) -> None:
        """Time update for **every** mixture hypothesis."""
        new_hyps: list[list[BernoulliComponent]] = []
        for comps in self._pmbm_hypothesis_components:
            pred: list[BernoulliComponent] = []
            for c in comps:
                cc = _clone_bernoulli_like(c)
                cc.predict()
                cc.r = float(self.survival_prob) * float(cc.r)
                pred.append(cc)
            new_hyps.append(pred)
        self._pmbm_hypothesis_components = new_hyps

    def prune_and_merge(self) -> None:
        """Prune low-*r* components in **each** hypothesis; assign track IDs on MAP only."""
        pruned_hyps: list[list[BernoulliComponent]] = []
        for comps in self._pmbm_hypothesis_components:
            pc = [c for c in comps if c.r > self.pruning_threshold]
            pc = [c for c in pc if not (c.age > 50 and c.r < 0.3)]
            pruned_hyps.append(pc)
        self._pmbm_hypothesis_components = pruned_hyps

        if len(self._pmbm_log_weights) != len(self._pmbm_hypothesis_components):
            self._pmbm_log_weights = np.zeros(
                len(self._pmbm_hypothesis_components), dtype=np.float64
            )
        if len(self._pmbm_hypothesis_components) == 0:
            self._pmbm_log_weights = np.array([0.0], dtype=np.float64)
            self._pmbm_hypothesis_components = [[]]
            self.bernoulli_components = []
            return

        imap = int(np.argmax(self._pmbm_log_weights))
        self.bernoulli_components = deepcopy(self._pmbm_hypothesis_components[imap])

        for component in self.bernoulli_components:
            if (
                component.r > self.existence_threshold
                and component.track_id is None
                and component.detection_count >= self.min_track_length
            ):
                component.track_id = self.next_track_id
                self.next_track_id += 1

        self._pmbm_hypothesis_components[imap] = deepcopy(self.bernoulli_components)

    # --- cost matrix + assignment variants --------------------------------------

    def _build_textbook_cost_matrix(
        self,
        components: list[BernoulliComponent],
        detections: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        BIG = 1.0e9
        M = len(components)
        N = len(detections)
        ncol = N + M
        C = np.full((M, ncol), BIG, dtype=np.float64)
        likelihoods = np.zeros((M, N), dtype=np.float64)
        gated = np.zeros((M, N), dtype=bool)

        for i, component in enumerate(components):
            for j, det in enumerate(detections):
                m2 = _mahalanobis_sq_position(component, det["centroid"])
                if m2 <= self.textbook_mahalanobis_gate_sq:
                    L = float(component.likelihood(det["centroid"]))
                    likelihoods[i, j] = max(L, 1e-30)
                    gated[i, j] = True
                    C[i, j] = -np.log(likelihoods[i, j])
            C[i, N + i] = self._textbook_miss_association_cost()

        return C, likelihoods, gated, M, N

    def _enumerate_assignment_variants(
        self, C: np.ndarray, M: int, N: int
    ) -> list[tuple[float, dict[int, int]]]:
        """Optimal Hungarian + a **small** set of forced-miss alternatives.

        For each parent hypothesis, running ``M+1`` full assignments scales
        terribly when ``M`` is large (dozens of extra `linear_sum_assignment`
        calls **per frame per parent**). We only perturb rows that took a
        **detection** in the optimal solution, and cap perturbations at
        ``pmbm_k_best - 1``.
        """
        BIG = 1.0e9
        results: list[tuple[float, dict[int, int]]] = []
        seen: set[tuple[int, ...]] = set()

        def pack_cols(row_ind: np.ndarray, col_ind: np.ndarray) -> dict[int, int]:
            return {int(r): int(c) for r, c in zip(row_ind.tolist(), col_ind.tolist())}

        def add(row_ind: np.ndarray, col_ind: np.ndarray) -> None:
            d = pack_cols(row_ind, col_ind)
            key = tuple(d.get(i, N + i) for i in range(M))
            cost = float(C[row_ind, col_ind].sum())
            if key not in seen:
                seen.add(key)
                results.append((cost, d))

        r0, c0 = linear_sum_assignment(C)
        add(r0, c0)

        max_alt = max(0, self.pmbm_k_best - 1)
        matched_rows = [int(i) for i, ci in enumerate(c0) if int(ci) < N]
        for r_miss in matched_rows[:max_alt]:
            C2 = C.copy()
            C2[r_miss, :N] = BIG
            r, c = linear_sum_assignment(C2)
            add(r, c)

        results.sort(key=lambda x: x[0])
        return results[: self.pmbm_k_best]

    def _apply_textbook_assignment(
        self,
        components: list[BernoulliComponent],
        detections: list,
        col_for_row: dict[int, int],
        likelihoods: np.ndarray,
        gated: np.ndarray,
        C: np.ndarray,
        N: int,
        M: int,
    ) -> list[BernoulliComponent]:
        BIG = 1.0e9
        used_det_cols: set[int] = set()
        updated_components: list[BernoulliComponent] = []

        for i, component in enumerate(components):
            ccol = col_for_row.get(i, N + i)
            if ccol >= N:
                missed = _clone_bernoulli_like(component)
                missed.r = (1.0 - self.detection_prob) * component.r
                missed.consecutive_detections = 0
                updated_components.append(missed)
                continue

            j = int(ccol)
            if not gated[i, j] or C[i, j] >= BIG * 0.5:
                missed = _clone_bernoulli_like(component)
                missed.r = (1.0 - self.detection_prob) * component.r
                missed.consecutive_detections = 0
                updated_components.append(missed)
                continue

            L = likelihoods[i, j]
            r_new = self._r_update_associated(component, L)
            comp_copy = _clone_bernoulli_like(component)
            comp_copy.update(
                detections[j]["centroid"], detections[j].get("intensity", 0.0)
            )
            comp_copy.r = r_new
            updated_components.append(comp_copy)
            used_det_cols.add(j)

        r0 = self._poisson_birth_existence()
        for j, det in enumerate(detections):
            if j in used_det_cols:
                continue
            pos = det["centroid"]
            state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
            P = np.eye(4, dtype=np.float64) * 100.0
            born = BernoulliComponent(r0, state, P, track_id=None)
            born.update(pos, det.get("intensity", 0.0))
            updated_components.append(born)

        return updated_components

    def _births_from_all_detections(self, detections: list) -> list[BernoulliComponent]:
        r0 = self._poisson_birth_existence()
        out: list[BernoulliComponent] = []
        for det in detections:
            pos = det["centroid"]
            state = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
            P = np.eye(4, dtype=np.float64) * 100.0
            born = BernoulliComponent(r0, state, P, track_id=None)
            born.update(pos, det.get("intensity", 0.0))
            out.append(born)
        return out

    def _miss_all_components(
        self, components: list[BernoulliComponent]
    ) -> list[BernoulliComponent]:
        out: list[BernoulliComponent] = []
        for component in components:
            missed = _clone_bernoulli_like(component)
            missed.r = (1.0 - self.detection_prob) * component.r
            missed.consecutive_detections = 0
            out.append(missed)
        return out

    def update_components(self, detections: list) -> None:
        """Branch mixture over association variants; normalize log-weights."""
        n_parents = len(self._pmbm_hypothesis_components)
        if n_parents != len(self._pmbm_log_weights):
            self._pmbm_log_weights = np.zeros(n_parents, dtype=np.float64)

        child_log_ws: list[float] = []
        child_hyps: list[list[BernoulliComponent]] = []

        for p in range(n_parents):
            log_w = float(self._pmbm_log_weights[p])
            comps = self._pmbm_hypothesis_components[p]
            n_c = len(comps)
            n_d = len(detections)

            if n_c == 0:
                if n_d == 0:
                    child_log_ws.append(log_w)
                    child_hyps.append([])
                else:
                    born = self._births_from_all_detections(detections)
                    child_log_ws.append(log_w)
                    child_hyps.append(born)
                continue

            if n_d == 0:
                child_log_ws.append(log_w)
                child_hyps.append(self._miss_all_components(comps))
                continue

            C, likelihoods, gated, M, N = self._build_textbook_cost_matrix(
                comps, detections
            )
            variants = self._enumerate_assignment_variants(C, M, N)
            if not variants:
                continue
            best_cost = variants[0][0]
            for cost, col_map in variants:
                delta = cost - best_cost
                lw_child = log_w - delta
                out_comps = self._apply_textbook_assignment(
                    comps, detections, col_map, likelihoods, gated, C, N, M
                )
                child_log_ws.append(lw_child)
                child_hyps.append(out_comps)

        if not child_hyps:
            self._pmbm_log_weights = np.array([0.0], dtype=np.float64)
            self._pmbm_hypothesis_components = [[]]
            self.bernoulli_components = []
            return

        lw = np.array(child_log_ws, dtype=np.float64)
        lw -= logsumexp(lw)

        # Keep top hypotheses by weight
        order = np.argsort(-lw)
        cap = min(self.pmbm_max_hypotheses, len(order))
        order = order[:cap]
        lw = lw[order]
        hyps = [child_hyps[i] for i in order.tolist()]
        lw -= logsumexp(lw)

        self._pmbm_log_weights = lw
        self._pmbm_hypothesis_components = hyps

        imap = int(np.argmax(self._pmbm_log_weights))
        self.bernoulli_components = deepcopy(self._pmbm_hypothesis_components[imap])

    def save_track_log(self, track_log: list) -> None:
        with open(self.track_log_path, "w", encoding="utf-8") as f:
            f.write(
                "# Poisson Multi-Bernoulli Mixture (PMBM-style) log — MAP hypothesis tracks\n"
            )
            f.write(f"# Input: {self.input_video_path}\n")
            f.write(f"# Mixture: up to {self.pmbm_max_hypotheses} hypotheses, "
                    f"{self.pmbm_k_best} assignment branches per parent\n")
            f.write("#\n")
            f.write(
                "# frame, track_id, x, y, existence_prob, confidence, speed, age, detections\n"
            )
            f.write("#" + "=" * 80 + "\n")
            for entry in track_log:
                f.write(
                    f"{entry['frame']}, {entry['track_id']}, "
                    f"{entry['x']}, {entry['y']}, "
                    f"{entry['existence_prob']:.4f}, {entry['confidence']:.4f}, "
                    f"{entry['speed']:.4f}, {entry['age']}, {entry['detections']}\n"
                )
        if self.verbose:
            print(f"   📝 Track log saved: {self.track_log_path}")
