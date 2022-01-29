import logging


import os
import json
import numpy as np
import random
import random
import torch
import random
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForPreTraining,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset


logging.basicConfig(
    format="%(asctime)s — %(levelname)s — %(name)s — %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


MODEL = "gpt2-medium"
SPECIAL_TOKENS = {
    "bos_token": "<|BOS|>",
    "eos_token": "<|EOS|>",
    "unk_token": "<|UNK|>",
    "pad_token": "<|PAD|>",
    "sep_token": "<|SEP|>",
}
MAXLEN = 768
TRAIN_SIZE = 0.8
EPOCHS = 4
LR = 5e-4
EPS = 1e-8
WARMUP_PROPORTION = 0.1
UNFREEZE_LAST_N = 6

SEED = 54321

TRAIN_BATCHSIZE = 1
BATCH_UPDATE = 16
# fp16 only useful in batch_size > 8
seed_everything(SEED)
TOTAL_SIZE = 1000  # 20000
np
dataset = []
data_path = "/content/drive/MyDrive/code/emailwriter/data/messages.jsonl"  # "data/messages.jsonl"
with open(data_path) as f:
    for line in f:
        message = json.loads(line)["message"]
        if len(message.split()) > 15:
            dataset.append(message)
random.shuffle(dataset)
dataset = dataset[:TOTAL_SIZE]
train = dataset[: int(len(dataset) * TRAIN_SIZE)]
test = dataset[int(len(dataset) * TRAIN_SIZE) :]
warmup_steps = int(len(train) / TRAIN_BATCHSIZE * WARMUP_PROPORTION)


def get_tokenier(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print("Special tokens added")
    return tokenizer


def get_model(tokenizer, special_tokens=None, load_model_path=None):

    # GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(
            MODEL,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            output_hidden_states=False,
        )
    else:
        config = AutoConfig.from_pretrained(MODEL, pad_token_id=tokenizer.eos_token_id, output_hidden_states=False)

    # ----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        # Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model


class EmailDataset(Dataset):
    def __init__(self, data, tokenizer, randomize=True):

        self.randomize = randomize
        self.tokenizer = tokenizer
        self.data = data

    # ---------------------------------------------#

    def __len__(self):
        return len(self.data)

    # ---------------------------------------------#

    def __getitem__(self, i):
        text = self.data[i]

        # input = (
        #     SPECIAL_TOKENS["bos_token"]
        #     + self.title[i]
        #     + SPECIAL_TOKENS["sep_token"]
        #     + kw
        #     + SPECIAL_TOKENS["sep_token"]
        #     + self.text[i]
        #     + SPECIAL_TOKENS["eos_token"]
        # )

        encodings_dict = self.tokenizer(text, truncation=True, max_length=MAXLEN, padding="max_length")

        input_ids = encodings_dict["input_ids"]
        attention_mask = encodings_dict["attention_mask"]

        return {
            "label": torch.tensor(input_ids),
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


########################################################################################################################

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForPreTraining.from_pretrained(MODEL, output_hidden_states=False).to(device)

train_dataset = EmailDataset(train, tokenizer)
val_dataset = EmailDataset(test, tokenizer, randomize=False)

# - Freeze selective layers:
# - Freeze all layers except last n:
for parameter in model.parameters():
    parameter.requires_grad = False

for i, m in enumerate(model.transformer.h):
    # Only un-freeze the last n transformer blocks
    if i + 1 > 12 - UNFREEZE_LAST_N:
        for parameter in m.parameters():
            parameter.requires_grad = True

for parameter in model.transformer.ln_f.parameters():
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():
    parameter.requires_grad = True


training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/code/emailwriter/models",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=TRAIN_BATCHSIZE,
    per_device_eval_batch_size=TRAIN_BATCHSIZE,
    eval_accumulation_steps=10,
    save_steps=int(TRAIN_BATCHSIZE) / 2,
    evaluation_strategy="steps",
    eval_steps=int(TRAIN_BATCHSIZE) / 2,
    warmup_steps=warmup_steps,
    learning_rate=LR,
    adam_epsilon=EPS,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    dataloader_num_workers=4,
)

# ---------------------------------------------------#
trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer
)

# ---------------------------------------------------#
trainer.train()
trainer.save_model()
