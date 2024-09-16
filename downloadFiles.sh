#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if 'data' directory does not exist and then create it
if [[ ! -e $DIR/data ]]; then
    mkdir "$DIR/data"
else
    echo "'data' directory already exists."
fi

# Download the traffic_analysis.mov file from Google Drive
gdown -O "$DIR/data/vehiclesTraffic1.mp4" "https://drive.google.com/uc?id=1vhWHMxg0pe-FXYOssJ3jsIWkRqcq8cid"
gdown -O "$DIR/data/vehiclesTraffic2.mov" "https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W"

# Download the traffic_analysis.pt file from Google Drive
# gdown -O "$DIR/data/vehiclesTraffic.pt" "model"