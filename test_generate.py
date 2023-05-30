
from typing import List, Tuple, Dict, Callable, Any, Union
from functools import partial
from pathlib import Path
from copy import deepcopy
import json
import torch
from torch import Tensor, LongTensor, stack, flip, cat, full, argmax
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from miditok import REMI, REMIPlus
from torchtoolkit.data import create_subsets
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, GenerationConfig
from transformers.data.data_collator import DataCollatorMixin
from evaluate import load as load_metric
from miditok import REMI, MIDITokenizer
from miditok.constants import CHORD_MAPS
from miditoolkit import MidiFile
from tqdm import tqdm



DATASET_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data_composers")
TRAINING_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/training_remi")
TRAINING_PATH_BPE = Path("/Users/floriangardin/code/music/musiclang2/locals/data/training_remi_bpe")
TOKENIZER_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/tokenizer_remi.json")
MODEL_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/models_remi/")
MIDI_FILE_TEST = Path("/Users/floriangardin/code/music/musiclang2/locals/test.mid")

tokenizer = REMIPlus()
tokenizer.load_params(TOKENIZER_PATH)


generation_config = dict(
    max_new_tokens=500,  # extends samples by 512 tokens
    min_new_tokens=0,
    do_sample=True,     # but sample instead
    temperature=0.75,
    top_k=30,
    top_p=0.95,
    #epsilon_cutoff=3e-4,
    #eta_cutoff=1e-3,
    pad_token_id=tokenizer['PAD_None'],
)


model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

model.eval()
inputs = torch.tensor([[0]], dtype=torch.long)
tokens = model.generate(inputs, **generation_config)[0]  # (N,T)
midi = tokenizer.tokens_to_midi(deepcopy(tokens), time_division=384)
midi.dump(MIDI_FILE_TEST)