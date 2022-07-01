<div align="center">

# Email Efficiency: Readability

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

</div>

## Description

This project is to take an email the format of which is a string of words and output a score to represent the readability of that email.

## Model Structure

This project builds and trains a linear regression model consisting of two layers: a `distilbert-base-uncased` layer and a `Dense` layer. Specifically, this model employs `distilbert-base-uncased` to consume each input email, and output the embedding of token `CLS` to represent the whole input email, then the `Dense` layer takes this embedding and outputs the final score.

## Data

The raw training data is called `CLEAR_corpus_final.xlsx`, `utils/email_process.py` is to clean the raw data. 

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/zhiyuanpeng/EmailEfficiency
cd EmailEfficiency

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash

# train on GPU
python main.py --expname 'your experiment name'
```

## Save and Load Model

This section describes how to convert `.pt` model to `.onnx` model for less inference latency. After training, a `.pt` model is saved in path `checkpoint`. Run `export_to_onnx.py` to achieve the conversion. `CHECKPOINT` is the path of the `.pt` model to be converted, and `MODEL_FILE` is the path of converted `.onnx` model. The details are as follows:

```bash
    # create_model is to convert the .pt model to .onnx model, it must take a real input for the
    # conversion process to know how to optimize the model sturcture for .onnx model
    create_model(input_ids_xx, attention_mask_yy)

    # after conversion, we can load test data to do inference
    t_trues, t_preds = [], []
    # load data
    test_data = pd.read_parquet(join(DATA_DIR, "email/4724_data_new_score/processed/test.parquet"))
    # load model
    session = create_session(MODEL_FILE)
    for _, row in test_data.iterrows():
        email = row["Excerpt"]
        email_embedding = tokenizer(email, padding=True, truncation=True, max_length=512)
        t_trues.append(float(row["Lexile Band"]))
        input_ids_x = np.array(email_embedding["input_ids"]).reshape(1, -1)
        attention_mask_y = np.array(email_embedding["attention_mask"]).reshape(1, -1)
        # z is the outputed score
        z = session.run(["z"], {"x":input_ids_x, "y":attention_mask_y})[0].reshape(-1)[0]*1000
        t_preds.append(z)
    # run the metric to evaluate the model
    test_loss = rmse(t_preds, t_trues)
    print(test_loss)
```

## Model Performance

Root Mean Square Error (RMSE) is used to evaluate the model which can be computed as:

![equation](https://latex.codecogs.com/svg.image?RMSE%20=%20%5Csqrt%7B%5Cfrac%7B%5Csum_%7Bi=1%7D%5E%7Bi=N%7D(y_%7Bi%7D-y_%7Bi%7D%5E%7Bpred%7D)%5E%7B2%7D%7D%7BN%7D%7D)

where:

![equation](https://latex.codecogs.com/svg.image?y_%7Bi%7D) is the true score for email ![equation](https://latex.codecogs.com/svg.image?i).

![equation](https://latex.codecogs.com/svg.image?y_%7Bi%7D%5E%7Bpred%7D) is the predicted score for email ![equation](https://latex.codecogs.com/svg.image?i).

![equation](https://latex.codecogs.com/svg.image?N) is the total number of emails.

On test data, RMSE is 80.0.
