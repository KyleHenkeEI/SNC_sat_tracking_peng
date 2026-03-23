param(
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r "$PSScriptRoot\requirements.txt"
