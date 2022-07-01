import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

class EmailDataset(Dataset):
    def __init__(self, df, bert_name: str = "distilbert-base-uncased"):
        """
        create dataset for dataloader
        Args:
            df: 
            bert_name: 
        """
        super().__init__()
        self.df = df
        self.bert_name = bert_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)

    def __getitem__(self, index):
        return (
            self.df.iloc[index]["text"].lower(), 
            self.df.iloc[index]["label"]
        )

    def __len__(self):
        return len(self.df)

    def collate_func(self, data_list):
        email, score = zip(*data_list)
        email_embedding = self.tokenizer(list(email), padding=True, truncation=True, return_tensors="pt", max_length=512)
        score = torch.LongTensor(score)
        return email_embedding, score