#!/bin/bash

MODEL_PATH="/opt/offroad_traversability/models/deeplabv3_mnv3_finetuned.onnx"

if [ "$#" -ne 2 ]; then
    echo "Usage: offroad_run.sh <input_video> <output_video>"
    exit 1
fi

offroad_segmentation "$MODEL_PATH" "$1" "$2"
