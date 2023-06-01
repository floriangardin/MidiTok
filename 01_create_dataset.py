import os

from miditok import REMI, REMIPlus
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from miditok.constants import ADDITIONAL_TOKENS

from miditok.config import Config

config: Config = Config("config.json")
CONTINUE = True
# Creates the tokenizer and loads a MIDI
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

# Converts MIDI files to tokens saved as JSON files
midi_paths = list(Path(config.dataset_path).glob("**/*.mid"))


existing_files = set(os.listdir(config.tokens_path))
# Remove files with filenames in existing_files

midi_paths = [midi_path for midi_path in midi_paths if midi_path.stem + ".json" not in existing_files]

#tokenizer.tokenize_midi_dataset(midi_paths, config.tokens_path)

# Constructs the vocabulary with BPE, from the tokenized files
tokenizer.learn_bpe(
    vocab_size=3000,
    tokens_paths=list(Path(config.tokens_path).glob("**/*.json")),
    start_from_empty_voc=False,
)

# Saving our tokenizer, to retrieve it back later with the load_params method
tokenizer.save_params(config.tokenizer_path)

# Converts the tokenized musics into tokens with BPE
tokenizer.apply_bpe_to_dataset(config.tokens_path, config.tokens_bpe_path)