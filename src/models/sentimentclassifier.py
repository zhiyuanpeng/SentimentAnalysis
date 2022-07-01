import os
from os.path import join
from typing import Any, List

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from torch import nn
from transformers import AutoTokenizer
from src.models.components.bert_embedding import BertEmbedding
import textstat


class SentimentClassifier(nn.Module):
    """Example of LightningModule for Race Classifier. A LightningModule organizes your PyTorch
    code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        bert_name: str = "pparasurama/raceBERT",
        dr: float = 0.1,
        lr: float = 5e-5,
        lr_decay: float = 0.0,
        adam_eps: float = 1e-8,
        warmup_proportion: float = 0.1,
        weight_decay=0.01,
        feature_mode: int = 0,
        **kwargs,
    ):
        """[summary]
        Args:
            bert_name (str, optional): [description]. Defaults to "bert-base-cased".
            dr (float, optional): [description]. Defaults to 0.1.
            lr (float, optional): [description]. Defaults to 5e-5.
            lr_decay (float, optional): [description]. Defaults to 0.0.
            adam_eps (float, optional): [description]. Defaults to 1e-8.
            warmup_proportion (float, optional): [description]. Defaults to 0.1.
            weight_decay (float, optional): [description]. Defaults to 0.01.
            fine_tune (bool, optional): [description]. Defaults to True.
            feature_mode (int, optional): [description]. Defaults to 0.
                1: extract the embedding of CLS
                2: average the embeddings of all pieces
        """
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.bert_name = bert_name
        self.dr = dr
        self.lr = lr
        self.lr_decay = lr_decay
        self.adam_eps = adam_eps
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay
        self.feature_mode = feature_mode
        #
        self.model = BertEmbedding(self.bert_name, self.feature_mode)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.dropout = nn.Dropout(p=self.dr)
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch

        # labels
        self.classifier = nn.Sequential(
            nn.Linear(768, 1),
            nn.Sigmoid()
        )
        self.loss_funct = nn.BCELoss()
        # self.loss_funct = nn.CrossEntropyLoss()

    def predict(self, email: str):
        """
        used for inference
        Args:
            x:
        Returns:
        """
        email_embedding = self.tokenizer(email, padding=True, truncation=True, return_tensors="pt", max_length=512)
        email_embedding = {k: v.cuda() for k, v in email_embedding.items()}
        valid_output = self.model(
            email_embedding["input_ids"], 
            email_embedding["attention_mask"],
        )
        logits = self.classifier(valid_output)
        # logits (batch_size, len(labels))
        return logits

    def forward(self, input_ids, attention_mask):
        valid_output = self.model(
            input_ids,
            attention_mask,
        )
        logits = self.classifier(valid_output)
        # logits (batch_size, len(labels))
        return logits