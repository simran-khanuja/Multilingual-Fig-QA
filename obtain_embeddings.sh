#!/bin/bash

OUTPUT_DIR = "/projects/metis2/anubha/research/Multilingual-Fig-QA/translated_data_for_embeddings/embeddings"

python obtain_embeddings.py \
    --model_name_or_path xlm-roberta-large \
    --input_file "translated_data_for_embeddings/translated_metaphor.tsv"\
    --max_length 128 \
    --per_device_eval_batch_size 32 \
    --output_dir "/projects/metis2/anubha/research/Multilingual-Fig-QA/translated_data_for_embeddings/embeddings" \
    --seed 0 \
    --use_labse