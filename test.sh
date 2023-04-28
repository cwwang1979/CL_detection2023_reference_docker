#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="8g"

docker volume create cldetection_alg_2023-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm --gpus all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/:/input/images/lateral-dental-x-rays/ \
        -v cldetection_alg_2023-output-$VOLUME_SUFFIX:/output/ \
        cldetection_alg_2023



docker run --rm \
        -v cldetection_alg_2023-output-$VOLUME_SUFFIX:/output/ \
        python:3.9-slim cat /output/orthodontic-landmarks.json | python -m json.tool

docker run --rm \
        -v cldetection_alg_2023-output:/output/ \
        -v $SCRIPTPATH/test/:/input/images/lateral-dental-x-rays/ \
        python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/orthodontic-landmarks.json')); f2 = json.load(open('/input/images/lateral-dental-x-rays/expected_output.json')); sys.exit(f1 != f2);"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm dldetection_alg_2023-output-$VOLUME_SUFFIX
