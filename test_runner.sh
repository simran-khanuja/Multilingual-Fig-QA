#!/bin/bash

# We currently have a bunch of shell scripts that run the experiments. 
# This basically provides an easier interface to access all experiments and run multiple times

SEEDS=( 10 20 30 40 50 )

print_help() {
    echo "Usage: $0 -e <experiment setting> -o <output file> -m <model name> "
    echo -e "\t-e: Experiment setting. One of [train, zero2hero]"
    echo -e "\t-o: Output file. The file to write the results to"
    echo -e "\t-m: Model name. One of [xlm-roberta-large, xlm-roberta-base, bert-base-multilingual-cased]"
    exit 1
}

while getopts "e:o:m:" opt; do
    case $opt in
        e)
            EXPERIMENT=$OPTARG
            ;;
        o)
            OUTPUT_FILE=$OPTARG
            ;;
        m)
            MODEL=$OPTARG
            ;;
        \?)
            print_help
            ;;
    esac
done

if [ -z "$EXPERIMENT" ] || [ -z "$OUTPUT_FILE" ] || [ -z "$MODEL" ]; then
    print_help
fi

if [ "$MODEL" == "xlm-roberta-large" ]; then
    LR=5e-6
    BATCH_SIZE=32
    NUM_EPOCHS=20
elif [ "$MODEL" == "xlm-roberta-base" ]; then
    LR=2e-5
    BATCH_SIZE=64
    NUM_EPOCHS=30
elif [ "$MODEL" == "bert-base-multilingual-cased" ]; then
    LR=5e-5
    BATCH_SIZE=64
    NUM_EPOCHS=30
else
    echo "Invalid model name"
    exit 1
fi

if [ "$EXPERIMENT" == "train" ]; then
    for SEED in "${SEEDS[@]}"; do
        echo "Seed: ${SEED}"
        ./run_baselines_train.sh "${OUTPUT}" "${MODEL}" "${SEED}" "${LR}" "${BATCH_SIZE}" "${EPOCHS}"
        ./run_baselines_test.sh "${OUTPUT}" "${LR}" "${SEED}"
    done
elif [ "$EXPERIMENT" == "zero2hero" ]; then
    for SEED in "${SEEDS[@]}"; do
        # TODO: test this one
        echo "Seed: ${SEED}"
        ./run_baselines_zero2hero.sh "${OUTPUT}" "${MODEL}" "${SEED}" "${LR}" "${BATCH_SIZE}" "${EPOCHS}"
        ./run_baselines_test.sh "${OUTPUT}" "${LR}" "${SEED}"
    done
else
    print_help
fi