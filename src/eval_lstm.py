import torch
import evaluate

rouge = evaluate.load("rouge")


def evaluate_model(model, dataset, vocab, max_new_tokens=5, max_samples=200):
    device = next(model.parameters()).device
    idx2word = {v: k for k, v in vocab.items()}
    all_predictions = []
    all_references = []
    for x, y in dataset:
        if len(all_predictions) >= max_samples:
            break
        n = len(x)
        cut = int(n * 0.75)
        prefix = x[:cut]
        target = x[cut:]
        if len(prefix) == 0 or len(target) == 0:
            continue
        input_ids = prefix.unsqueeze(0).to(device)
        lengths = torch.tensor([len(prefix)])
        generated = model.generate(input_ids, lengths, max_new_tokens=max_new_tokens)
        new_tokens = generated[0][len(prefix) :]
        predicted_text = " ".join([idx2word[tok.item()] for tok in new_tokens])
        target_text = " ".join([idx2word[tok.item()] for tok in target])
        all_predictions.append(predicted_text)
        all_references.append(target_text)

    results = rouge.compute(predictions=all_predictions, references=all_references)
    return results
