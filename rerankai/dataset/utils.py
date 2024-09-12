
import math
import os
import random
from dataclasses import dataclass
import time
from typing import List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding

from rerankai.arguments import DataArguments
from sklearn.model_selection import train_test_split

def create_label_data(batch_data):
    labels = torch.zeros(batch_data.shape[0], dtype=torch.long)
    return labels

def create_batch_data(batch_data):
    batch_data_ = batch_data["input_ids"].contiguous().view(batch_data["input_ids"].shape[0]*batch_data["input_ids"].shape[1],
                                                            batch_data["input_ids"].shape[2])
    mask_data_ = batch_data["attention_mask"].contiguous().view(batch_data["attention_mask"].shape[0]*batch_data["attention_mask"].shape[1],
                                                                batch_data["attention_mask"].shape[2])
    return {
        'input_ids': batch_data_,
        'attention_mask': mask_data_,
        'labels':create_label_data(batch_data["input_ids"])
    }



def pad_sequence(sequence,mask, max_length):
    seq_length = sequence.shape[1]
    padded_sequence = torch.zeros((sequence.shape[0], max_length), dtype=sequence.dtype)
    attention_mask = torch.zeros((sequence.shape[0], max_length), dtype=mask.dtype)
    
    padded_sequence[:, :seq_length] = sequence
    attention_mask[:, :seq_length] = mask
    
    return padded_sequence, attention_mask

def collate_fn(batch):
    # 获取当前batch中每个样本的最大长度
    max_length = max(item[0].shape[1] for item in batch)
    
    input_ids_batch = []
    attention_mask_batch = []
    
    for item in batch:
        input_ids_padded, attention_mask_padded = pad_sequence(item[0], item[1],max_length)
        input_ids_batch.append(input_ids_padded)
        attention_mask_batch.append(attention_mask_padded)
    
    input_ids_batch = torch.stack(input_ids_batch)
    attention_mask_batch = torch.stack(attention_mask_batch)
    
    return {
        'input_ids': input_ids_batch,
        'attention_mask': attention_mask_batch
    }


@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)

class TrainDatasetForCE(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            train=True,  
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        train_test_splits = self.dataset.train_test_split(test_size=1 - 0.7, seed=42)
    
        self.dataset = train_test_splits['train'] if train else train_test_splits['test']

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        self.group_collator = GroupCollator(tokenizer)

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']
        pos = random.choice(self.dataset[item]['pos'])
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)

        batch_data = []
        batch_data.append(self.create_one_example(query, pos))
        for neg in negs:
            batch_data.append(self.create_one_example(query, neg))

        group_batch_data = self.group_collator(batch_data)
        
        return group_batch_data["input_ids"], group_batch_data["attention_mask"]

