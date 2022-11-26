# TEST
test_files=( "dev_fr_en.csv" "dev_fr_fr.csv" "dev_hi_en.csv" "dev_hi_hi.csv" "dev_ja_en.csv" "dev_ja_ja.csv" )
languages=( "fr_en" "fr_fr" "hi_en" "hi_hi" "ja_en" "ja_ja" )

LR=5e-6
SEED=10
MODEL=xlm_roberta_large

for index in "${!test_files[@]}"
do
    rm -rf "./${MODEL}/embeddings_seed${SEED}_lr${LR}_${languages[index]}"
    mkdir -p "./${MODEL}/embeddings_seed${SEED}_lr${LR}_${languages[index]}"
    echo "======================================================================="
    echo "========== Metaphor ${MODEL} LR ${LR} seed ${SEED} language ${languages[index]} ========"
    echo "======================================================================="
    python run_baselines.py \
    --model_name_or_path ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
    --test_file translated_dev_sets/${test_files[index]} \
    --do_predict \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 20 \
    --output_dir ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
    --seed $SEED \
    --save_embeddings \
    --save_embeddings_in_tsv \
    --embedding_output_dir ./${MODEL}/embeddings_seed${SEED}_lr${LR}_${languages[index]}
done