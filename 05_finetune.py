import os

from miditok import REMI, REMIPlus
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from miditok.constants import ADDITIONAL_TOKENS

from miditok.config import Config


from miditok.funcs import create_dataset, prepare_dataset

"""
Finetuning is keeping the same tokenizer, but create a new tokenized dataset.

"""
if __name__ == "__main__":
    config: Config = Config("config.json")

    # Load all arguments and update the config
    print('Step 1: Create dataset')
    #
    dataset_path = config.finetuning_path
    tokens_path = config.finetuned_tokens_path
    tokens_bpe_path = config.finetuned_tokens_bpe_path
    tokenizer_path = config.tokenizer_path
    #
    # create_dataset(dataset_path, tokens_path, tokens_bpe_path, tokenizer_path, finetune=True)

    print('Step 2: Prepare dataset')

    # Prepare dataset
    tokens_split_path = config.finetuned_tokens_split_path

    prepare_dataset(tokens_bpe_path, tokenizer_path, tokens_split_path)