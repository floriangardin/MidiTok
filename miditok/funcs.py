import os
from miditok import REMI, REMIPlus
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from miditok.constants import ADDITIONAL_TOKENS
import glob
import random
import numpy as np
from tqdm import tqdm
import json

def create_dataset(dataset_path, tokens_path, tokens_bpe_path, tokenizer_path, finetune=False):
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
    try:
        existing_files = set(os.listdir(tokens_path))
    except Exception as e:
        existing_files = set()

    if finetune:
        tokenizer.load_params(tokenizer_path)
        # Converts MIDI files to tokens saved as JSON files
        midi_paths = list(Path(dataset_path).glob("**/*.mid"))
        # Remove files with filenames in existing_files
        midi_paths = [midi_path for midi_path in midi_paths if midi_path.stem + ".json" not in existing_files]
        tokenizer.tokenize_midi_dataset(midi_paths, tokens_bpe_path)

    if not finetune:
        # Converts MIDI files to tokens saved as JSON files
        midi_paths = list(Path(dataset_path).glob("**/*.mid"))
        # Remove files with filenames in existing_files
        midi_paths = [midi_path for midi_path in midi_paths if midi_path.stem + ".json" not in existing_files]
        tokenizer.tokenize_midi_dataset(midi_paths, tokens_path)

        files = list(Path(tokens_path).glob("**/*.json"))
        import random
        random.shuffle(files)
        kept_files = files[:1000]

        # Constructs the vocabulary with BPE, from the tokenized files
        tokenizer.learn_bpe(
            vocab_size=3000,
            tokens_paths=kept_files,
            start_from_empty_voc=False,
        )

        # Saving our tokenizer, to retrieve it back later with the load_params method
        tokenizer.save_params(tokenizer_path)

        # Converts the tokenized musics into tokens with BPE
        tokenizer.apply_bpe_to_dataset(tokens_path, tokens_bpe_path)



def prepare_dataset(tokens_bpe_path, tokenizer_path, tokens_split_path):



    all_bpe_files = glob.glob(tokens_bpe_path + "/*.json")
    if not os.path.exists(tokens_split_path):
        os.makedirs(tokens_split_path)
    print('Starting...')
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
    tokenizer.load_params(tokenizer_path)

    EOS_TOKEN = tokenizer['EOS_None']

    # Shuffle train test in all_bpe_files
    random.shuffle(all_bpe_files)
    split = 0.99
    train_files = all_bpe_files[:int(len(all_bpe_files) * split)]
    val_files = all_bpe_files[int(len(all_bpe_files) * split):]

    # We add EOS token to each file
    train_ids = []
    for file in tqdm(train_files, desc="Loading train files"):
        train_ids.append(json.load(open(file, 'r'))['ids'] + [EOS_TOKEN])

    val_ids = []
    for file in tqdm(val_files, desc="Loading val files"):
        val_ids.append(json.load(open(file, 'r'))['ids'] + [EOS_TOKEN])

    # Shuffle
    len_train = sum([len(train_id) for train_id in train_ids], 0)
    len_val = sum([len(val_id) for val_id in val_ids], 0)

    filename = os.path.join(tokens_split_path, "train.bin")
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(len_train,))
    idx = 0
    for train_id in tqdm(train_ids, desc="Saving train ids"):
        arr[idx:idx + len(train_id)] = train_id
        idx += len(train_id)
    arr.flush()

    print("Training size : ", len(arr))
    filename = os.path.join(tokens_split_path, "val.bin")
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(len_val,))
    idx = 0
    for val_id in tqdm(val_ids, desc="Saving train ids"):
        arr[idx:idx + len(val_id)] = val_id
        idx += len(val_id)
    arr.flush()

    print("Val size : ", len(arr))
    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')

