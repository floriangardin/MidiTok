import os

from miditok import REMI, REMIPlus
from miditok.utils import get_midi_programs
from miditoolkit import MidiFile
from pathlib import Path
from miditok.constants import ADDITIONAL_TOKENS
from miditok.dataset import MIDIDataset, DataCollatorGen
from torchtoolkit.data import create_subsets
from transformers import GPT2LMHeadModel, GPT2Config, Trainer, TrainingArguments, GenerationConfig
from torch import Tensor, LongTensor, stack, flip, cat, full, argmax
from evaluate import load as load_metric

DATASET_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data_composers")
TRAINING_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/training_remi")
TRAINING_PATH_BPE = Path("/Users/floriangardin/code/music/musiclang2/locals/data/training_remi_bpe")
TOKENIZER_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/tokenizer_remi.json")
MODEL_PATH = Path("/Users/floriangardin/code/music/musiclang2/locals/data/models_remi/")
os.makedirs(MODEL_PATH, exist_ok=True)

# PARAMS
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
FP16 = False

tokenizer = REMIPlus()
tokenizer.load_params(TOKENIZER_PATH)


tokens_paths = list(TRAINING_PATH_BPE.glob("**/*.json"))
dataset = MIDIDataset(
    tokens_paths, max_seq_len=512, min_seq_len=384,
)
subset_train, subset_valid = create_subsets(dataset, [0.3])



config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=512,
    n_embd=384,
    n_layer=4,
    n_head=4,
    padding_token_id=tokenizer['PAD_None'],
    bos_token_id=tokenizer['BOS_None'],
    eos_token_id=tokenizer['EOS_None'],
)
model = GPT2LMHeadModel(config)


metrics = {metric: load_metric(metric) for metric in ["accuracy"]}

def compute_metrics(eval_pred):
    """Computes metrics for pretraining.
    Must use proprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    predictions, labels = eval_pred
    not_pad_mask = labels != -100
    labels, predictions = labels[not_pad_mask], predictions[not_pad_mask]
    return metrics["accuracy"].compute(predictions=predictions.flatten(), references=labels.flatten())

def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """Preprocesses the logits before accumulating them during evaluation.
    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids

training_config = TrainingArguments(
    MODEL_PATH, True, True, True, False, "steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    eval_accumulation_steps=None,
    eval_steps=1000,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=3.0,
    max_steps=2000,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
    #no_cuda=False,
    seed=444,
    fp16=FP16,
    load_best_model_at_end=True,
    label_smoothing_factor=0.,
    optim="adamw_torch",
    #report_to=["tensorboard"],
    gradient_checkpointing=True,
)

trainer = Trainer(
    model=model,
    args=training_config,
    data_collator=DataCollatorGen(tokenizer["PAD_None"]),
    train_dataset=subset_train,
    eval_dataset=subset_valid,
    compute_metrics=compute_metrics,
    callbacks=None,
    preprocess_logits_for_metrics=preprocess_logits,
)

# Training
train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()