
from configs.config import *
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter



def clean_data(data):
    data = data.copy()
    data["text"] = data["text"].str.lower()
    data["text"] = data["text"].str.replace(r"http\S+", "", regex=True)
    data["text"] = data["text"].str.replace(r"@\S+", "", regex=True)
    data["text"] = data["text"].str.replace(r"[^a-z\s]", "", regex=True)
    data["text"] = data["text"].str.replace(r"\s+", " ", regex=True)
    data["text"] = data["text"].str.strip()
    data = data[data["text"].str.len() > 0]
    return data


def load_and_clean(path):
    data = pd.read_csv(path, header=None, names=["text"], sep="\x01")
    cleaned_data = clean_data(data)
    cleaned_data.to_csv(PREPROCESSED_PATH, index=False)
    return cleaned_data


def prepare_dataset(tokens):
    X = tokens[:-1]
    y = tokens[1:]
    return X, y


def tokenize(text):
    return text.split()


def split_data(data):
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=RANDOM_STATE)

    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    return train_df, val_df, test_df


def build_vocab(train_df, max_vocab_size=None, min_freq=2):
    counter = Counter()
    for text in train_df["text"]:
        for word in tokenize(text):
            counter[word] += 1
    
    filtered = [(w, c) for w, c in counter.items() if c >= min_freq]
    filtered.sort(key=lambda x: -x[1])
    if max_vocab_size:
        filtered = filtered[:max_vocab_size]
    
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
    for idx, (word, _) in enumerate(filtered, start=3):
        vocab[word] = idx
    return vocab