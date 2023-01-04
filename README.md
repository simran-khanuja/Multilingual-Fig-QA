# Multilingual-Fig-QA
Creating the multilingual version of the [Fig-QA](https://arxiv.org/pdf/2204.12632.pdf) dataset.

### Running experiments
Experiments can be run through `test_runner.sh`.
Recommend making the output directory name different from the huggingface model name as it may cause problems with huggingface otherwise.
Results will be output to a file named `results.txt` or `z2h_results.txt`.

#### Multilingual training

`./test_runner.sh -e train -o <output_dir> -m <model_name>`

<model_name> can be `bert-base-multilingual-cased`, `xlm-roberta-base` or `xlm-roberta-large`.

#### Multilingual training

`./test_runner.sh -e zero2hero -o <output_dir> -m <model_name>`
