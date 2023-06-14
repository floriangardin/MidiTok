import json
from miditok import REMIPlus
from pathlib import Path
import os
from tqdm import tqdm
import random
import numpy as np
import glob
from miditok.config import Config
import os
from miditok.constants import ADDITIONAL_TOKENS
from miditok.funcs import prepare_dataset

config: Config = Config("config.json")
tokens_bpe_path = config.tokens_bpe_path
tokenizer_path = config.tokenizer_path
tokens_split_path = config.tokens_split_path


prepare_dataset(tokens_bpe_path, tokenizer_path, tokens_split_path)
