#!/bin/bash
MODEL=xlm_roberta_large
MODEL_NAME_OR_PATH=xlm-roberta-large
LR=5e-6
SEED=10
declare -a languages=( "hi" "id" "su" "sw" "jv" )
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
    python run_baselines.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_file /projects/metis2/anubha/research/Multilingual-Fig-QA/data/addition_data_winogrand/train/${languages[lang]}/${languages[lang]}_${number[num]}.csv  \
    --validation_file /projects/metis2/anubha/research/Multilingual-Fig-QA/langdata/en_dev.csv \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 20 \
    --output_dir ./${MODEL}/${languages[lang]}_${number[num]}/ckpts_seed${SEED}_lr${LR} \
    --seed $SEED
    done
done