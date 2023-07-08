#!/bin/bash

cmd=$1

# constants
IMAGE_NAME="model_predictor"
IMAGE_TAG=$(git describe --always)

if [[ -z "$cmd" ]]; then
    echo "Missing command"
    exit 1
fi

run_predictor() {
    model_1_config_path=$1
    model_2_config_path=$2
    port=$3

    if [[ -z "$model_1_config_path" ]]; then
        echo "Missing model_config_path"
        exit 1
    fi
    if [[ -z "$port" ]]; then
        echo "Missing port"
        exit 1
    fi

    docker build -f platform/model_predictor/Dockerfile -t $IMAGE_NAME:$IMAGE_TAG .
    IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG \
        MODEL_1_CONFIG_PATH=$model_1_config_path MODEL_2_CONFIG_PATH=$model_2_config_path \
        PORT=$port \
        docker-compose -f platform/model_predictor/docker-compose.yml up -d
}

shift

case $cmd in
run_predictor)
    run_predictor "$@"
    ;;
*)
    echo -n "Unknown command: $cmd"
    exit 1
    ;;
esac
