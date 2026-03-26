import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out

    def generate(self, input_ids, lengths, max_new_tokens=10, eos_token_id=2):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids, lengths)
                last_token = outputs[:, -1, :]
                predicted_token = last_token.argmax(dim=-1)
                if predicted_token.item() == eos_token_id:
                    break
                input_ids = torch.cat([input_ids, predicted_token.unsqueeze(1)], dim=1)
                lengths = lengths + 1

        return input_ids
