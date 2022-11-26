MODEL=xlm_roberta_large
MODEL_NAME_OR_PATH=xlm-roberta-large

# TRAIN
for LR in 5e-6
do
	for SEED in 10
	do
	echo "======================================================================="
	echo "========================= Metaphor ${MODEL} LR ${LR} seed ${SEED} ==========================="
	echo "======================================================================="
	python run_baselines.py \
	--model_name_or_path ${MODEL_NAME_OR_PATH} \
	--train_file /home/skhanuja/Fig-QA/data/filtered/train.csv \
    --validation_file /home/skhanuja/Fig-QA/data/filtered/dev.csv \
	--max_length 128 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--gradient_accumulation_steps 1 \
	--learning_rate ${LR} \
	--num_train_epochs 20 \
	--output_dir ./${MODEL}/ckpts_seed${SEED}_lr${LR} \
	--seed $SEED
	done
done
