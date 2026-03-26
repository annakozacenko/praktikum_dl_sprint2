import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from configs.config import *
import torch
from tqdm import tqdm
from lstm_model import LSTMModel
from next_token_dataset import NextTokenDataset, collate_fn
from torch.utils.data import DataLoader
from data_utils import split_data, build_vocab, load_and_clean
from eval_lstm import evaluate_model

os.makedirs("models", exist_ok=True)


import csv

with open("training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "rouge1", "rouge2"])


data = load_and_clean("data/raw_dataset.txt")
train_df, val_df, test_df = split_data(data)

    #train_df= train_df.head(10000)
    #val_df = val_df.head(1000)

vocab = build_vocab(train_df, MAX_VOCAB_SIZE, MIN_FREQ)

train_dataset = NextTokenDataset(train_df, vocab)
val_dataset = NextTokenDataset(val_df, vocab)


batch_size = BATCH_SIZE

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
)

vocab_size = len(vocab)
embedding_dim = EMBEDDING_DIM
hidden_size = HIDDEN_SIZE
output_size = len(vocab)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем: {device}")

model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, output_size, NUM_LAYERS, DROPOUT)


model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
n_epochs = EPOCHS

for epoch in range(n_epochs):
    model.train()
    total_train_loss, total_val_loss = 0.0, 0.0
    for batch in tqdm(train_dataloader):
        inputs, labels, lengths = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    model.eval()
    for batch in tqdm(val_dataloader):
        inputs, labels, lengths = batch

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs, lengths)
            loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_dataloader)
    #rouge_results = evaluate_model(model, val_dataset, vocab)
    sample_texts = val_df["text"].tolist()
    rouge_results = evaluate_model(model, sample_texts, vocab)
    print(
        f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"ROUGE-1: {rouge_results['rouge1']:.4f}, "
        f"ROUGE-2: {rouge_results['rouge2']:.4f}"
    )
    with open("training_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, rouge_results['rouge1'], rouge_results['rouge2']])


torch.save(model.state_dict(), MODEL_PATH)
print("Модель сохранена в models/lstm_model.pt")


import json
with open(VOCAB_PATH, "w") as f:
    json.dump(vocab, f)

print("Модель и словарь сохранены")