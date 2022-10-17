# TEST
test_files=( "dev_fr_en.csv" "dev_fr_fr.csv" "dev_hi_en.csv" "dev_hi_hi.csv" "dev_ja_en.csv" "dev_ja_ja.csv" )
LR=5e-6
SEED=10
MODEL=xlmr_large

for test_file in "${test_files[@]}"
do
    echo "======================================================================="
    echo "========================= Metaphor ${MODEL} LR ${LR} seed ${SEED} ==========================="
    echo "======================================================================="
    python run_baselines.py \
    --model_name_or_path ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
    --train_file train.csv \
    --validation_file dev.csv \
    --test_file translated_dev_sets/${test_file} \
    --do_predict \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 20 \
    --output_dir ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
    --seed $SEED > ./logs/${MODEL}_seed${SEED}_lr${LR}_${test_file}.log
done