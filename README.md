# Sorghum-Leaf
## Requirements

numpy
scipy
scikit-image
opencv
networkx
globus_sdk
gdal
utm
## Description 
The description is [here](doc/length_documentation.md)
## Usage
```
usage: run-pipeline [-h] --raw-path RAW_PATH --ply-path PLY_PATH -o
                    OUTPUT_PATH --start START --end END [--no-download]
                    [--scanner {east,west,both}] [--crop] [-v] [-p PROCESSES]                   

optional arguments:
  -h, --help            show this help message and exit
  --raw-path RAW_PATH   path to the root of raw data
  --ply-path PLY_PATH   path to the root of ply data
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        yyyy-mm-dd
  --start START         Start date. Format: yyyy-mm-dd
  --end END             End date. Format: yyyy-mm-dd
  --no-download         no download ply files from globus
  --scanner {east,west,both}
                        from which scanner
  --crop                by plot or by leaf
  -v, --verbose
  -p PROCESSES, --processes PROCESSES
                        number of sub-processes
```
