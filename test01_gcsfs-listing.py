#!/usr/bin/env python

import gcsfs

fs = gcsfs.GCSFileSystem(project='Sierra Nevada Corp')
# all_files = fs.ls('ei_snc_data/data/STARLINK_30254/03/31/00')
# all_files = fs.glob('ei_snc_data/data/STARLINK_30254/03/31/*/*image*')
all_files = fs.glob('ei_snc_data/data/STARLINK_30254/03/*/??/0-??????0-0-image.rawl*')
print(all_files)
print(len(all_files))

