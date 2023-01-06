MODEL=${1:-xlm_roberta_large}
MODEL_NAME_OR_PATH=${2:-xlm-roberta-large}
SEED_ARG=${3:-10}
LR_ARG=${4:-5e-6}
BATCH_SIZE_ARG=${5:-32}
NUM_EPOCHS_ARG=${6:-20}

# TRAIN
for LR in ${LR_ARG}
do
	for SEED in ${SEED_ARG}
	do
	echo "======================================================================="
	echo "========================= TRAIN: Metaphor ${MODEL} LR ${LR} seed ${SEED} ==========================="
	echo "======================================================================="
	python run_baselines.py \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_file ./langdata/en_train.csv \
    --validation_file ./langdata/en_dev.csv \
	--max_length 128 \
	--per_device_train_batch_size ${BATCH_SIZE_ARG} \
	--per_device_eval_batch_size 32 \
	--gradient_accumulation_steps 1 \
	--learning_rate ${LR} \
	--num_train_epochs ${NUM_EPOCHS_ARG} \
	--output_dir ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
	--seed $SEED
	done
done