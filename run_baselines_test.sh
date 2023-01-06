# TEST
test_files=( "hi.csv" "id.csv" "jv.csv" "kn.csv" "su.csv" "sw.csv" )
languages=( "hi" "id" "jv" "kn" "su" "sw" )

LR=${2:-5e-6}
SEED=${3:-10}
MODEL=${1:-xlm_roberta_large}

for index in "${!test_files[@]}"
do
    rm -rf "./${MODEL}/embeddings_seed${SEED}_lr${LR}_${languages[index]}"
    mkdir -p "./${MODEL}/embeddings_seed${SEED}_lr${LR}_${languages[index]}"
    echo "======================================================================="
    echo "========== Metaphor ${MODEL} LR ${LR} seed ${SEED} language ${languages[index]} ========"
    echo "======================================================================="
    python run_baselines.py \
    --model_name_or_path ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
--test_file langdata/${test_files[index]} \
    --do_predict \
    --max_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${LR} \
    --num_train_epochs 20 \
    --output_dir ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
    --seed $SEED \
    --test_runner_mode \
    --save_embeddings \
    --save_embeddings_in_tsv \
    --embedding_output_dir ./${MODEL}/embeddings_seed${SEED}_lr${LR}_${languages[index]}
done