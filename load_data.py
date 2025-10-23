import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle
import sqlite3

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch
from typing import List, Tuple 

TOKENIZER_ID = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_ID)
PAD_IDX = tokenizer.pad_token_id

class T5Dataset(Dataset):

    def __init__(self, data_folder: str, split: str):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.x, self.y = self.process_data(data_folder, split)
        self.schema_str = _schema_string(os.path.join(data_folder, "flight_database.db"))

    def process_data(self, data_folder: str, split: str) -> Tuple[List[str], List[str] | None]:
        nl_path = os.path.join(data_folder, f"{split}.nl")
        x = load_lines(nl_path)

        sql_path = os.path.join(data_folder, f"{split}.sql")
        if os.path.exists(sql_path) and split != "test":
            y = load_lines(sql_path)
        else:
            y = None
        return x, y
    
    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        # ----- encoder text: clear instruction + (optional) schema -----
        # T5 works best with a task-style prefix.
        prefix = "translate natural language to SQL:"
        if getattr(self, "schema_str", ""):
            encoder_text = f"{prefix} NL: {self.x[idx]} Schema: {self.schema_str}"
        else:
            encoder_text = f"{prefix} NL: {self.x[idx]}"
    
        enc = tokenizer(encoder_text, truncation=True, max_length=512, return_tensors=None)
        enc_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
    
        # test split has no targets
        if self.split == "test":
            return {"encoder_ids": enc_ids}
    
        # ----- targets -----
        tgt = tokenizer(self.y[idx], truncation=True, max_length=256, return_tensors=None)
        y_ids = torch.tensor(tgt["input_ids"], dtype=torch.long)
    
        # T5: decoder typically starts with PAD; labels are the unshifted target ids
        bos = torch.tensor([PAD_IDX], dtype=torch.long)
        dec_in = torch.cat([bos, y_ids[:-1]], dim=0) if y_ids.numel() > 0 else bos.clone()
        dec_tgt = y_ids
    
        return {
            "encoder_ids": enc_ids,
            "decoder_inputs": dec_in,
            "decoder_targets": dec_tgt,
        }

def _pad_batch(seqs: List[torch.Tensor]) -> torch.Tensor:
    return pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX)

def _schema_string(db_path: str) -> str:
    if not os.path.exists(db_path):
        return ""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    tables = [r[0] for r in c.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()]
    pieces = []
    for t in sorted(tables):
        cols = [r[1] for r in c.execute(f"PRAGMA table_info({t});").fetchall()]
        pieces.append(f"{t}(" + ", ".join(cols) + ")")
    conn.close()
    return " | ".join(pieces)

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encs = [b["encoder_ids"] for b in batch]
    dec_ins = [b["decoder_inputs"] for b in batch]
    dec_tgts = [b["decoder_targets"] for b in batch]

    encoder_ids = _pad_batch(encs)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    decoder_inputs = _pad_batch(dec_ins)
    decoder_targets = _pad_batch(dec_tgts)

    # First token at decoding time (used by some training loops)
    initial_decoder_inputs = torch.full((encoder_ids.size(0), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encs = [b["encoder_ids"] for b in batch]
    encoder_ids = _pad_batch(encs)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.full((encoder_ids.size(0), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(data_folder: str, batch_size: int, test_batch_size: int):
    train_set = T5Dataset(data_folder, "train")
    dev_set = T5Dataset(data_folder, "dev")
    test_set = T5Dataset(data_folder, "test")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=normal_collate_fn, pin_memory=True, num_workers=1)
    dev_loader = DataLoader(dev_set, batch_size=test_batch_size, shuffle=False, collate_fn=normal_collate_fn, pin_memory=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, collate_fn=test_collate_fn, pin_memory=True, num_workers=1)
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines()]

def load_prompting_data(data_folder: str):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x