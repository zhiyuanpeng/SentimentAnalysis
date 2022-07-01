import torch
from torch import nn
from src.datamodules.email_datamodule import EmailDataModule
from src.models.sentimentclassifier import SentimentClassifier
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

WORK_DIR = os.getcwd()
DATA_DIR = join(WORK_DIR, "data")
#BERT_NAME = join(DATA_DIR, "Bert/distilbert-base-uncased")
BERT_NAME = "distilbert-base-uncased"

def configure_optimizers(model):
    no_decay = ["bias", "LayerNorm.weight"]
    model_params = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model_params if not any(nd in n for nd in no_decay)],
            "weight_decay": model.weight_decay,
        },
        {
            "params": [p for n, p in model_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # return torch.optim.AdamW(
    #     params=self.parameters(), lr=self.lr,
    #     eps=self.adam_eps
    # )
    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=model.lr,
        eps=model.adam_eps,
    )

def train(args, logger, writer, checkpoint):
    # build dataset
    email = EmailDataModule(
            bert_name=BERT_NAME,
            batch_size=args.batch_size,
            num_workers=4,
            data_dir=join(DATA_DIR, "email/imdb_clean/processed"))
    email.setup()
    # build model
    model = SentimentClassifier(
            bert_name=BERT_NAME,
            dr=0.1,
            lr=args.lr,
            lr_decay=0.0,
            adam_eps=1e-8,
            warmup_proportion=0.1,
            weight_decay=0.01,
            feature_mode = 1,
            ).cuda()
    # loss_funct = nn.CrossEntropyLoss()
    loss_funct = nn.BCELoss()
    #sig = nn.Sigmoid()
    # build optimizer
    optimizer = configure_optimizers(model)
    # train
    step_count = 0
    best_f1 = -float("inf")
    for step in tqdm(range(args.epochs)):
        model.train()
        count = 0
        for email_embedding, score in tqdm(email.train_dataloader()):
            email_embedding = {k: v.cuda() for k, v in email_embedding.items()}
            logits = model(email_embedding["input_ids"], email_embedding["attention_mask"])
            # loss = torch.sqrt(loss_funct(logits, score.cuda()))
            loss = loss_funct(logits.view(-1), score.to(torch.float32).cuda())
            writer.add_scalar('train/loss', loss, step_count)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            step_count += 1
            if step_count % 100 == 0:
                model.eval()
                v_trues, v_preds = [], []
                with torch.no_grad():
                    for email_embedding, score in tqdm(email.val_dataloader()):
                        email_embedding = {k: v.cuda() for k, v in email_embedding.items()}
                        preds = model(email_embedding["input_ids"], email_embedding["attention_mask"])
                        v_trues.extend(score.cuda())
                        v_preds.extend((preds>0.5)*1)
                f1 = f1_score([i.detach().cpu().item() for i in v_trues], [i.detach().cpu().item() for i in v_preds], average="micro")
                report = classification_report([i.detach().cpu().item() for i in v_trues], [i.detach().cpu().item() for i in v_preds], labels=[0,1])
                print(report)
                print("epoch", step, ", the valid f1 is ", f1)
                writer.add_scalar('valid/f1', f1, step_count)
                #logger.info(f"*****************{step_count}********************")
                # save best checkpoint
                if f1 > best_f1:
                    best_f1 = f1
                    if os.path.exists(checkpoint):
                        os.remove(checkpoint)
                    #logger.info(f"old checkpoint is removed")
                    torch.save(model.state_dict(), checkpoint)
            if step_count > 305:
                break
    model.eval()
    #logger.info(f"loading model from checkpoint {checkpoint}")
    model.load_state_dict(torch.load(checkpoint))
    t_trues, t_preds = [], []
    with torch.no_grad():
        for email_embedding, score in tqdm(email.test_dataloader()):
            email_embedding = {k: v.cuda() for k, v in email_embedding.items()}
            preds = model(email_embedding["input_ids"], email_embedding["attention_mask"])
            t_trues.extend(score.cuda())
            t_preds.extend((preds>0.5)*1)
        test_f1 = f1_score([i.detach().cpu().item() for i in t_trues], [i.detach().cpu().item() for i in t_preds], average="micro")
        test_report = classification_report([i.detach().cpu().item() for i in t_trues], [i.detach().cpu().item() for i in t_preds], labels=[0,1])
        print("Test model on test dataset:\n")
        print(test_report)
        print("the final test f1 is ", f1)

            


def main():
    parser = argparse.ArgumentParser(description='PyTorch Uncertainty Training')
    #Network
    parser.add_argument('--expname', default='debug', type=str, help='exp name')
    parser.add_argument('--lr', default=2e-5, type=float, help='learning_rate')
    parser.add_argument('--batch_size', default=24, type=int, help='batch_size')
    parser.add_argument('--epochs', default=1, type=int, help='batch_size')
    #Summary
    args = parser.parse_args()
    # degine logger
    log_dir = join(f"{WORK_DIR}", f"logs/{args.expname}/")
    #log_dir = WORK_DIR+'/logs/'+args.expname+'/'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    logging.basicConfig(filename=join(log_dir, "log.txt"),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    logging.info("Running")
    logger = logging.getLogger(__name__)
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    seed_everything(logger, seed=345)
    checkpoint = join(log_dir, 'model1.pt')
    train(args, logger, writer, checkpoint)

if __name__ == "__main__":
    main()
