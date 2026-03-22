import torch
from torch.utils.data import DataLoader, Dataset


from data_utils import prepare_dataset, tokenize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class NextTokenDataset(Dataset):
    def __init__(self, data, vocab):
        self.samples = []
        for text in data["text"]:
            tokens = tokenize(text)
            token_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
            if len(token_ids) < 2:
                continue
            X, y = prepare_dataset(token_ids)
            self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    lengths = torch.tensor([len(item) for item in x])
    padded_texts = pad_sequence(x, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(y, batch_first=True, padding_value=0)
    return padded_texts, padded_labels, lengths
