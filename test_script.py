from miditok import REMIPlus, REMI,  constants
from miditoolkit import Instrument, Note, TempoChange, TimeSignature, MidiFile

from miditok import REMI, MIDILike, Octuple, TSD
from miditok import REMIPlus
from miditok.constants import ADDITIONAL_TOKENS
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path

test_mid = '/Users/floriangardin/code/music/musiclang2/locals/test.mid'
timedivision = 480
midi = MidiFile()
midi.time_signature_changes = [TimeSignature(6, 8, 0)]
midi.tempo_changes = [TempoChange(120.,0)]
inst = Instrument(0, 0, 'piano')

# NOTE: add
inst.notes = [Note(60, pitch=64, start=i*timedivision, end=(i+1)*timedivision) for i in range(8)]
midi.instruments.append(inst)
tokenizer = REMIPlus(
        additional_tokens={
            **ADDITIONAL_TOKENS,
            "Chord": True,
            "chord_tokens_with_root_note": True,
            "Program": True,
            "Tempo": True,
            "TimeSignature": True,
        },
        max_bar_embedding=None,
        beat_res={(0, 8): 16}
    )
tokenized = tokenizer(midi)
print(tokenized.tokens)
converted_back_midi = tokenizer(tokenized)  # PyTorch / Tensorflow / Numpy tensors supported
converted_back_midi.dump(test_mid)

#exit()


#path = "/Users/floriangardin/code/music/musiclang2/locals/lmd_full/0/0a2b1c146699c22ec10f5cd008b20ee1.mid"
path = "/Users/floriangardin/code/music/musiclang2/locals/data_composers/chopin/chpn-p1.mid"
tokenizer = REMIPlus(
        additional_tokens={
            **ADDITIONAL_TOKENS,
            "Chord": True,
            "chord_tokens_with_root_note": True,
            "Program": True,
            "Tempo": True,
            "TimeSignature": True,
        },
        max_bar_embedding=None,
        beat_res={(0, 8): 16}
    )
midi = MidiFile(path)

tokenized = tokenizer(midi)
print(tokenized.tokens)
print(len(set(tokenized.tokens)))
# calling it will automatically detect MIDIs, paths and tokens before the conversion
converted_back_midi = tokenizer(tokenized)  # PyTorch / Tensorflow / Numpy tensors supported

converted_back_midi.dump(test_mid)
from pdb import set_trace; set_trace()