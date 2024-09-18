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
gdown -O "$DIR/data/vehiclesTraffic.pt" "https://drive.google.com/uc?id=1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z"

# Download the fraffic_analysis_overtrained.pt file from Google drive
gdown -O "$DIR/data/vehiclesTraffic_overtrained.pt" "https://drive.google.com/uc?id=1dajaD2vw32v2pfPNgIW2SP1RgNOe3Krv"