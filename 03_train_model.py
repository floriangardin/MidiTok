import os

import torch

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
from miditok.config import Config
from miditok.dataloader import DataLoader
import os
from transformers import TransfoXLConfig, TransfoXLModel


config: Config = Config("config.json")

# PARAMS

FP16 = torch.cuda.is_available()

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

if config.loading_method == "split":
    print('Using split dataloader')
    train_path = os.path.join(config.tokens_split_path, 'train.bin')
    val_path = os.path.join(config.tokens_split_path, 'val.bin')
    dataloader_train = DataLoader(train_path, block_size=config.model_config['n_positions'])
    dataloader_val = DataLoader(val_path, block_size=config.model_config['n_positions'], length=1000)
else:
    print('Using full dataloader')
    tokens_paths = list(Path(config.tokens_split_path).glob("**/*.json"))
    dataset = MIDIDataset(
        tokens_paths, max_seq_len=config.model_config['n_positions'], min_seq_len=384,
    )
    dataloader_train, dataloader_val = create_subsets(dataset, [0.3])


model_dict = {
    'gpt': GPT2LMHeadModel,
    'transfo-xl': TransfoXLModel,
}

if config.model_type == 'gpt':
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        padding_token_id=tokenizer['PAD_None'],
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer['EOS_None'],
        **config.model_config
    )
    model = GPT2LMHeadModel(model_config)

else:
    model_config = TransfoXLConfig(
        vocab_size=len(tokenizer),
        padding_token_id=tokenizer['PAD_None'],
        bos_token_id=tokenizer['BOS_None'],
        eos_token_id=tokenizer['EOS_None'],
        n_head=config.model_config['n_head'],
        n_layer=config.model_config['n_layer'],
        d_embed=config.model_config['n_embd'],
        cutoffs = []
    )
    model = TransfoXLModel(model_config)


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
    config.model_path, True, True, True, False, "steps",
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    eval_accumulation_steps=None,
    eval_steps=1000,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=3.0,
    max_steps=config.max_steps,
    lr_scheduler_type="cosine_with_restarts",
    warmup_ratio=0.3,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=3,
    #no_cuda=False,
    seed=444,
    fp16=FP16,
    load_best_model_at_end=True,
    label_smoothing_factor=0.,
    optim="adamw_torch",
    #report_to=["tensorboard"],
    gradient_checkpointing=config.model_type == 'gpt',
)

trainer = Trainer(
    model=model,
    args=training_config,
    train_dataset=dataloader_train,
    eval_dataset=dataloader_val,
    compute_metrics=compute_metrics,
    callbacks=None,
    preprocess_logits_for_metrics=preprocess_logits,
)

# Training
train_result = trainer.train(resume_from_checkpoint=config.resume)
trainer.save_model()  # Saves the tokenizer too
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()