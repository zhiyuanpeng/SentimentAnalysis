import os
from os import listdir
from unicodedata import decimal
import pandas as pd
from os.path import join, isfile
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import textstat
import numpy as np
import html
WORK_DIR = os.getcwd()
DATA_DIR = join(WORK_DIR, "data/email")

def clean_token(txt: str):
    return " ".join([c.lower() for c in txt.split() if c.isalnum()])

def len_check(from_path: str):
    """check the max len after tokenizer

    Args:
        from_path (str): _description_
    """
    df = pd.read_excel(from_path)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    for _, row in df.iterrows():
        tokens = tokenizer(row["Excerpt"])
        assert len(tokens.input_ids) <= 512
    print("check done!")

def load_raw_data(data_dir: str, is_global_zero: bool=True):
    """
    given the data dir of conll2003 dataset, return (train_raw, dev_raw,
    test_raw) where train_raw has [(sentence, label)] format
    Args:
        data_dir: the path to the train, valid and test
        label_list: for un leagal labels not in label_list, set "O"
        dataset_name: walmart use different format from bio format
        is_global_zero: only true print info, else not
    Returns:
        train_raw, dev_raw, test_raw
    """
    files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
    for file in files:
        if "train" in file:
            train_raw = pd.read_parquet(join(data_dir, file))
            if is_global_zero:
                print(f"train dataset has {len(train_raw)} rows")
        elif "valid" in file or "dev" in file:
            val_raw = pd.read_parquet(join(data_dir, file))
            if is_global_zero:
                print(f"dev dataset has {len(val_raw)} rows")
        elif "test" in file:
            test_raw = pd.read_parquet(join(data_dir, file))
            if is_global_zero:
                print(f"test dataset has {len(test_raw)} rows")
        else:
            continue
    return train_raw, val_raw, test_raw

def process(from_path: str, to_path: str):
    """extract col: Excerpt, Flesch-Reading-Ease,Flesch-Kincaid-Grade-Level,SMOG Readability

    Args:
        from_path (str): _description_
        to_path (str): _description_
    """
    #df = pd.read_excel(from_path, nrows=4692)
    df = pd.read_csv(from_path)
    #df = df[["Excerpt", "Lexile Band", "Flesch-Reading-Ease", "Flesch-Kincaid-Grade-Level", "SMOG Readability"]]
    df_train, df_test = train_test_split(df, train_size=0.9, test_size=0.1)
    df_train, df_valid = train_test_split(df_train, train_size=0.9, test_size=0.1)
    # to file
    df_train.to_parquet(join(to_path, "train.parquet"))
    df_valid.to_parquet(join(to_path, "valid.parquet"))
    df_test.to_parquet(join(to_path, "test.parquet"))

def process_update_score(from_path: str, to_path: str):
    """extract col: Excerpt, Flesch-Reading-Ease,Flesch-Kincaid-Grade-Level,SMOG Readability
    copy from process, use textstat

    Args:
        from_path (str): _description_
        to_path (str): _description_
    """
    #df = pd.read_excel(from_path, nrows=4692)
    df = pd.read_csv(from_path)
    # df = df[["Excerpt", "Lexile Band", "Flesch-Reading-Ease", "Flesch-Kincaid-Grade-Level", "SMOG Readability"]]
    df['text'] = df['text'].apply(clean_token)
    #df = df[["Excerpt", "Lexile Band"]]
    # df['Excerpt'].apply(remove_special)
    #flesch_readings, flesch_kincaids, smogs = [], [], []
    #for _, row in df.iterrows():
        #email = row["text"]
        #flesch_reading = textstat.flesch_reading_ease(email)
        #flesch_kincaid = textstat.flesch_kincaid_grade(email)
        #smog = textstat.smog_index(email)
        #
        #flesch_readings.append(np.round(flesch_reading, decimals=2))
        #flesch_kincaids.append(np.round(flesch_kincaid, decimals=2))
        #smogs.append(np.round(smog, decimals=2))
    #df["Flesch-Reading-Ease"] = flesch_readings
    #df["Flesch-Kincaid-Grade-Level"] = flesch_kincaids
    #df["SMOG Readability"] = smogs
    df_train, df_test = train_test_split(df, train_size=0.9, test_size=0.1)
    df_train, df_valid = train_test_split(df_train, train_size=0.9, test_size=0.1)
    # to file
    df_train.to_parquet(join(to_path, "train.parquet"))
    df_valid.to_parquet(join(to_path, "valid.parquet"))
    df_test.to_parquet(join(to_path, "test.parquet"))

def main():
    from_pt = join(DATA_DIR, "imdb/raw/movie.csv")
    to_pt = join(DATA_DIR, "imdb_clean/processed")
    # len_check(from_pt)
    # process(from_pt, join(DATA_DIR, to_pt))
    process_update_score(from_pt, join(DATA_DIR, to_pt))
    # txt = html.unescape('Hello, \n &nbsp; \n How are you are you doing fine and what is the thing that you want to be doing, this is a thing \n &nbsp;')
    # txt = "When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape.\nThe floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches.\nAt each end of the room, on the wall, hung a beautiful bear-skin rug.\nThese rugs were for prizes, one for the girls and one for the boys. And this was the game.\nThe girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole.\nThis would have been an easy matter, but each traveller was obliged to wear snowshoes."
    # clean_txt = clean_token(txt)
    print('done')

if __name__ == "__main__":
    main()