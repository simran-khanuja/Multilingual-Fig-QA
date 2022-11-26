#!/bin/bash

OUTPUT_DIR = labse_embeddings
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}

python obtain_embeddings.py \
    --model_name_or_path xlm-roberta-large \
    --input_file translated_metaphor.tsv \
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --output_dir ${OUTPUT_DIR} \
    --seed 0 \
    --use_labse