import os

from miditok import REMI, REMIPlus
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from miditok.constants import ADDITIONAL_TOKENS


TOKENIZER_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/tokenizer_remi.json")

EXAMPLE_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/training_remi_bpe/bach_846.json")

tokenizer = REMIPlus()
tokenizer.load_params(TOKENIZER_PATH)
tokens = tokenizer.load_tokens(EXAMPLE_PATH)
midi = tokenizer(tokens['ids'])


midi.dump("/Users/floriangardin/code/music/musiclang2/locals/test.mid")
from pdb import set_trace; set_trace()