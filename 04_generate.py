import os
from copy import deepcopy
import torch
from miditok import REMI, REMIPlus
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, GenerationConfig
from miditok.config import Config

config: Config = Config("config.json")
FROM_CHECKPOINT = True
CHECKPOINT = 15000

tokenizer = REMIPlus()
tokenizer.load_params(config.tokenizer_path)

generation_config = dict(
    max_new_tokens=500,  # extends samples by 512 tokens
    min_new_tokens=0,
    do_sample=True,     # but sample instead
    temperature=1.0,
    top_k=30,
    top_p=0.95,
    epsilon_cutoff=3e-4,
    eta_cutoff=1e-3,
    pad_token_id=tokenizer['PAD_None'],
)


if FROM_CHECKPOINT:
    model = GPT2LMHeadModel.from_pretrained(os.path.join(config.model_path, f'checkpoint-{CHECKPOINT}'))
else:
    model = GPT2LMHeadModel.from_pretrained(config.model_path)

model.eval()
import time
start =  time.time()
inputs = torch.tensor([[0]], dtype=torch.long)
tokens = model.generate(inputs, **generation_config)[0]  # (N,T)
print('Length of tokens:', len(tokens))
midi = tokenizer.tokens_to_midi(deepcopy(tokens), time_division=384)
midi.dump(config.midi_file_test)
print(f"Generated in {time.time() - start} seconds")