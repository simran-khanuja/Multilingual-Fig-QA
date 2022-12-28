import argparse
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

from run_baselines import prepare_data, main
import pdb

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
    if not args.use_grid_search:
        search_space = {
            "learning_rate": hp.uniform("learning_rate", 1e-7, 1e-5),
            "batch_size": hp.choice("batch_size", [8, 16, 32, 64]),
        }
    else:
        search_space = {
            "learning_rate": hp.choice("learning_rate", [1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]),
            "batch_size": hp.choice("batch_size", [8, 16, 32, 64])
        }
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    
    trials = Trials()
    best = fmin(objective, search_space, algo=tpe.suggest, max_evals=args["max_evals"], trials=trials)

    return space_eval(search_space, best), trials
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for hyperparameters")
    parser.add_argument("--train_file", type=str, default="langdata/en_train.csv")
    parser.add_argument("--eval_file", type=str, default="langdata/en_dev.csv")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased", choices=["bert-base-multilingual-cased", "xlm-roberta-base", "xlm-roberta-large", "facebook/xlm-roberta-xl"])
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_evals", type=int, default=100)
    parser.add_argument("--use_grid_search", action="store_true", help="Use grid search instead of hyperopt")
    args = parser.parse_args()
    
    print(f"finding hyperparams for {args.model_name} with {args.num_train_epochs} epochs")
    if args.use_grid_search:
        print("using grid search")

    MODEL_NAME = args.model_name
    NUM_EPOCHS = args.num_train_epochs
    MOCK_ARGS["num_train_epochs"] = NUM_EPOCHS
    MOCK_ARGS["train_file"] = args.train_file
    MOCK_ARGS["validation_file"] = args.eval_file

    best_config, trials = main_tune(args)
    print("DONE! Best hyperparameters: ", best_config)
    print("best loss: ", trials.best_trial["result"]["loss"])