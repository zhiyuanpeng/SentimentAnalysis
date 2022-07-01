import torch
from torch import nn
from src.datamodules.email_datamodule import EmailDataModule
from src.models.readability import Readability
from src.utils.torch_util import seed_everything
import argparse, sys
import numpy as np
import datetime
import shutil
from tqdm import tqdm
import os
from os.path import join
import logging
from sklearn.metrics import classification_report, f1_score
from torch.utils import tensorboard
import pandas as pd
from transformers import AutoTokenizer

def rmse(predictions, targets):
    predictions, targets = np.array(predictions), np.array(targets)
    return np.sqrt(np.mean((predictions-targets)**2))

WORK_DIR = os.getcwd()
DATA_DIR = join(WORK_DIR, "data")
LOG_DIR = join(WORK_DIR, "logs")
BERT_NAME = join(DATA_DIR, "Bert/distilbert-base-uncased")
CHECKPOINT = join(LOG_DIR, "epoch30_RMSE_decay0.01_clean_noextra/model.pth")
# CHECKPOINT = join(LOG_DIR, "debug/model.pth")
model = Readability(
        bert_name=BERT_NAME,
        feature_mode = 1,
    ).cuda()
loss_funct = nn.MSELoss()
model.eval()
model.load_state_dict(torch.load(CHECKPOINT))
t_trues, t_preds = [], []
# load data
test_data = pd.read_parquet(join(DATA_DIR, "email/4724_data_new_score/processed/test.parquet"))
with torch.no_grad():
    for _, row in test_data.iterrows():
        preds = model.predict(row["Excerpt"])
        t_trues.append(float(row["Lexile Band"]))
        t_preds.append(preds.item())
# test_loss = np.mean(test_loss)
test_loss = rmse(t_preds, t_trues)
print(f"*****************TEST RESULT********************")
print(f"The test loss is {test_loss}")