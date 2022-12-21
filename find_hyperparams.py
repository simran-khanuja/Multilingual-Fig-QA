import argparse
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

from run_baselines import prepare_data, main

NUM_EPOCHS = 20
MODEL_NAME = None

MOCK_ARGS = {
    "train_file": "langdata/en_train.csv",
    "validation_file": "langdata/en_dev.csv",
    "test_file": None,
    "config_name": None,
    "tokenizer_name": None,
    "use_slow_tokenizer": False,
    "do_predict": False,
    "save_embeddings": False,
    "save_embeddings_in_tsv": False,
    "max_length": 128,
    "pad_to_max_length": True,
    "lr_scheduler_type": "linear",
    "num_warmup_steps": 0,
    "output_dir": None,
    "seed": 10,
    "silent": True,
    "with_tracking": False,
    "push_to_hub": False,
    "push_to_hub_model_id": None,
    "source_lang": None,
    "target_lang": None,
    "dataset_name": None,
    "debug": False,
    "gradient_accumulation_steps": 1,
    "max_train_steps": None,
    "checkpointing_steps": None,
    "resume_from_checkpoint": None,
    "early_stopping_patience": None,
    "weight_decay": 0.0,
}

def objective(params):
    MOCK_ARGS["model_name_or_path"] = MODEL_NAME
    MOCK_ARGS["per_device_train_batch_size"] = params["batch_size"]
    MOCK_ARGS["per_device_eval_batch_size"] = params["batch_size"]
    MOCK_ARGS["learning_rate"] = params["learning_rate"]
    #MOCK_ARGS["weight_decay"] = params["weight_decay"]
    MOCK_ARGS["num_train_epochs"] = NUM_EPOCHS

    print(params)
    acc = main(MOCK_ARGS)
    return {"loss": -acc, "status": STATUS_OK}

def main_tune(args):
    search_space = {
        "learning_rate": hp.uniform("learning_rate", 1e-7, 1e-4),
        "batch_size": hp.choice("batch_size", [8, 16, 32, 64]),
    }
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    
    best = fmin(objective, search_space, algo=tpe.suggest, max_evals=10)

    print(best)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for hyperparameters")
    parser.add_argument("--train_file", type=str, default="langdata/en_train.csv")
    parser.add_argument("--eval_file", type=str, default="langdata/en_dev.csv")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", choices=["bert-base-multilingual-cased", "xlm-roberta-base", "xlm-roberta-large"])
    parser.add_argument("--num_train_epochs", type=int)
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    NUM_EPOCHS = args.num_train_epochs
    MOCK_ARGS["num_train_epochs"] = NUM_EPOCHS
    main_tune(args)