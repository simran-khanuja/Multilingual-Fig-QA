#!/bin/bash
MODEL=${1:-xlm_roberta_large_z2h}
MODEL_NAME_OR_PATH=${2:-xlm-roberta-large}
BASE_DIR=`pwd`
SEED=${3:-10}
LR=${4:-5e-6}
BATCH_SIZE=${5:-32}
declare -a languages=( "hi" "id" "jv" "kn" "su" "sw" )
declare -a number=( "2" "4" "6" "8" "10" )


# TRAIN
for lang in "${!languages[@]}"
do
    for num in "${!number[@]}"
    do
    rm -rf "./${MODEL}/${lang}_${num}/ckpts_seed${SEED}_lr${LR}"
    mkdir -p "./${MODEL}/${lang}_${num}/ckpts_seed${SEED}_lr${LR}"
    echo "======================================================================="
    echo "========== Metaphor ${MODEL} LR ${LR} seed ${SEED} language ${languages[lang]} number${number[num]} ========"
    echo "======================================================================="
    echo ${MODEL_NAME_OR_PATH}
    python run_baselines.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_file ${BASE_DIR}/data/sample_addition_data_winogrand/train/${languages[lang]}/${languages[lang]}_${number[num]}.csv  \
    --validation_file ${BASE_DIR}/langdata/en_dev.csv \
    --max_length 128 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 1 \
    --output_dir ./${MODEL}/${languages[lang]}_${number[num]}/ckpts_seed${SEED}_lr${LR} \
    --seed $SEED \
    --silent 
    done
done