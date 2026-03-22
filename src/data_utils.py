import pandas as pd
from sklearn.model_selection import train_test_split


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
    cleaned_data.to_csv("data/preprocessed_data.csv", index=False)
    return cleaned_data


def prepare_dataset(tokens):
    X = tokens[:-1]
    y = tokens[1:]
    return X, y


def tokenize(text):
    return text.split()


def split_data(data):
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)

    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    return train_df, val_df, test_df


def build_vocab(train_df):
    vocab = {"<unk>": 0}
    idx = 1
    for text in train_df["text"]:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab
