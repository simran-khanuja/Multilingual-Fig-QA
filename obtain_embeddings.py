import torch
"""
This script obtains [CLS] embeddings of sentences.
Input: Tsv file containing language\tsentence, per line.
Output: 
    - Embeddings in TSV format
    - Embeddings in tensor format

"""

import argparse
import logging
import os

import datasets
import torch

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    RobertaModel
)

from transformers import BertModel, BertTokenizerFast

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Obtain embeddings")
    parser.add_argument(
        "--input_file", type=str, default=None, help="A text file containing one sentence per line."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store embeddings.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--use_labse",
        action="store_true",
        help="If passed, use labse to obtain sentence embeddings.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    with open(args.input_file, "r") as f:
        sentences = []
        line = f.readline()
        while line:
            # Skip first line
            line = f.readline()
            lang_sent = line.split('\t')
            if len(lang_sent) == 2:
                sentences.append(lang_sent[1].strip())
            else:
                print("Skipping line: ", line)
            
    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
   
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    # Set output hidden states to True to get access to all hidden states
    config.output_hidden_states = True

    if args.use_labse:
        tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        model = BertModel.from_pretrained("setu4993/LaBSE")

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModel.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,)
    model.resize_token_embeddings(len(tokenizer))

    cls_embeddings_all = []

    tensor_outfile = os.path.join(args.output_dir, "embeddings.pt")
    tsv_outfile = os.path.join(args.output_dir, "embeddings.tsv")

    padding = "max_length" if args.pad_to_max_length else False

    for batch_index in range(0, len(sentences), args.per_device_eval_batch_size):
        sentence_batch = sentences[batch_index:batch_index + args.per_device_eval_batch_size]
        with torch.no_grad():
            inputs = tokenizer(sentence_batch, 
                                return_tensors="pt",
                                max_length=args.max_length,
                                padding=True,
                                truncation=True)
            outputs = model(**inputs)
            cls_embeddings = outputs.pooler_output
            cls_embeddings = accelerator.gather(cls_embeddings)
            if accelerator.is_main_process:
                for cls_embedding in cls_embeddings:
                    cls_embeddings_all.append(cls_embedding)
                    cls_embedding = cls_embedding.cpu().numpy()
                    with open(tsv_outfile, "a") as f:
                        f.write("\t".join([str(x) for x in cls_embedding]) + "\n")

    if accelerator.is_main_process:
        torch.save({"embeddings": cls_embeddings_all}, tensor_outfile)


if __name__ == "__main__":
    main()