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

config: Config = Config("config.json")

all_bpe_files = glob.glob(config.tokens_bpe_path + "/*.json")

tokenizer = REMIPlus()
tokenizer.load_params(config.tokenizer_path)
EOS_TOKEN = tokenizer['EOS_None']

# Shuffle train test in all_bpe_files
random.shuffle(all_bpe_files)
split = 0.98
train_files = all_bpe_files[:int(len(all_bpe_files) * split)]
val_files = all_bpe_files[int(len(all_bpe_files) * split):]

# We add EOS token to each file
train_ids  = [json.load(open(file, 'r'))['ids'] + [EOS_TOKEN] for file in train_files]
val_ids  = [json.load(open(file, 'r'))['ids'] + [EOS_TOKEN] for file in val_files]

# Shuffle
train_ids = sum(train_ids, [])
val_ids = sum(val_ids, [])


filename = os.path.join(config.tokens_split_path, "train.bin")
dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(len(train_ids),))
arr[:] = train_ids
arr.flush()

print("Training size : ", len(arr))

filename = os.path.join(config.tokens_split_path, "val.bin")
dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(len(val_ids),))
arr[:] = val_ids
arr.flush()

print("Val size : ", len(arr))
# train.bin is ~17GB, val.bin ~8.5MB
# train has ~9B tokens (9,035,582,198)
# val has ~4M tokens (4,434,897)

# to read the bin files later, e.g. with numpy:
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')

