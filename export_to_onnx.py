# A set of code samples showing different usage of the ONNX Runtime Python API
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import torch
import onnxruntime
import os
from os.path import join
from src.models.sentimentclassifier import SentimentClassifier, f1_score
from torch import nn
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer

WORK_DIR = os.getcwd()
DATA_DIR = join(WORK_DIR, "data")
LOG_DIR = join(WORK_DIR, "logs")
#BERT_NAME = join(DATA_DIR, "Bert/distilbert-base-uncased")
BERT_NAME = "distilbert-base-uncased"
CHECKPOINT = join(LOG_DIR, "debug/model1.pt")
MODEL_FILE = join(DATA_DIR,'onnx/sentiment.onnx')
DEVICE = torch.device("cpu")
# create model
model = SentimentClassifier(
        bert_name=BERT_NAME,
        feature_mode = 1,
    )
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE), strict=False)
model.eval()

# Create input
#test_data = pd.read_csv(join(DATA_DIR, "email/test/input.csv"), header=None, names = ['subject','Excerpt'])
test_data = pd.read_parquet(join(DATA_DIR, "email/imdb_clean/processed/test.parquet"))
email = test_data.iloc[0]["text"]
tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)
email_embedding = tokenizer(email, padding=True, truncation=True, return_tensors="pt", max_length=512)
# email_embedding = {k: v.cuda() for k, v in email_embedding.items()}
input_ids_xx, attention_mask_yy = email_embedding["input_ids"], email_embedding["attention_mask"]
# Create an instance of the model and export it to ONNX graph format, with dynamic size for the data
def create_model(input_ids, attention_mask):
    
    torch.onnx.export(
                        model, 
                        (input_ids, attention_mask), 
                        MODEL_FILE,
                        opset_version=11,
                        input_names=["x", "y"], 
                        output_names=["z"],
                        dynamic_axes={"x": {0: "batch_size", 1: "length"}, 
                                      "y": {0: "batch_size", 1: "length"}})
 
# Create an ONNX Runtime session with the provided model
def create_session(model: str) -> onnxruntime.InferenceSession:
    providers = ['CPUExecutionProvider']
    return onnxruntime.InferenceSession(model, providers=providers)

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    return tensor.cpu().numpy()

def rmse(predictions, targets):
    predictions, targets = np.array(predictions), np.array(targets)
    return np.sqrt(np.mean((predictions-targets)**2))

def main():
    # create_model is to convert the .pt model to .onnx model, it must take a real input for the
    # conversion process to know how to optimize the model sturcture for .onnx model
    create_model(input_ids_xx, attention_mask_yy)

    # after conversion, we can load test data to do inference
    t_trues, t_preds = [], []
    # load data
    #test_data = pd.read_csv(join(DATA_DIR, "email/test/input.csv"), header=None, names = ['subject','Excerpt'])
    test_data = pd.read_parquet(join(DATA_DIR, "email/imdb_clean/processed/test.parquet"))
    # load model
    session = create_session(MODEL_FILE)
    for _, row in test_data.iterrows():
        email = row["text"]
        email_embedding = tokenizer(email, padding=True, truncation=True, max_length=512)
        t_trues.append(float(row["label"]))
        input_ids_x = np.array(email_embedding["input_ids"]).reshape(1, -1)
        attention_mask_y = np.array(email_embedding["attention_mask"]).reshape(1, -1)
        # z is the outputed score
        z = session.run(["z"], {"x":input_ids_x, "y":attention_mask_y})[0].reshape(-1)[0]
        t_preds.append(z)
    # run the metric to evaluate the model
    preds = torch.tensor(t_preds, dtype=torch.float32)
    t_preds = (preds>0.5)*1
    print(t_trues)
    print(t_preds)
    print(classification_report(t_trues, t_preds))
    f1 = f1_score(t_trues, t_preds, average="micro")
    print("onnx model on test dataset f1: ", f1)

if __name__ == "__main__":
    main()   