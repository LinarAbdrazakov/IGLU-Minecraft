#!/bin/bash

docker run --rm -it --gpus all --network host --ipc host -v $(pwd)/../:/IGLU-Minecraft iglu_docker