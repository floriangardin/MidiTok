import os
from copy import deepcopy
import torch
from miditok import REMI, REMIPlus
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, GenerationConfig
from miditok.config import Config
from miditok.constants import ADDITIONAL_TOKENS


config: Config = Config("config.json")



tokenizer = REMIPlus(
additional_tokens={
            **ADDITIONAL_TOKENS,
            "Chord": False,
            "chord_tokens_with_root_note": False,
            "Program": True,
            "Tempo": True,
            "TimeSignature": True,
        },
        max_bar_embedding=None,
        beat_res={(0, 8): 16}
)
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


bar_token_id = tokenizer['Bar_None']
print(f"Bar token id: {bar_token_id}")

if config.use_checkpoint:
    model = GPT2LMHeadModel.from_pretrained(os.path.join(config.model_path, f'checkpoint-{config.checkpoint}'))
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
print(tokens)