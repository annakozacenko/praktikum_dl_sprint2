import torch
import evaluate

rouge = evaluate.load("rouge")


# def evaluate_model(model, dataset, vocab, max_new_tokens=5, max_samples=None):
#     device = next(model.parameters()).device
#     idx2word = {v: k for k, v in vocab.items()}
#     unk_id = vocab.get("<unk>", 1)
#     all_predictions = []
#     all_references = []

#     for x, y in dataset:
#         if len(all_predictions) >= max_samples:
#             break
#         n = len(x)
#         cut = int(n * 0.75)
#         prefix = x[:cut]
#         target = x[cut:]
#         if len(prefix) == 0 or len(target) == 0:
#             continue
#         input_ids = prefix.unsqueeze(0).to(device)
#         lengths = torch.tensor([len(prefix)])
#         generated = model.generate(input_ids, lengths, max_new_tokens=max_new_tokens)
#         new_tokens = generated[0][len(prefix) :]
#         predicted_text = " ".join([idx2word[tok.item()] for tok in new_tokens])
#         target_text = " ".join([idx2word[tok.item()] for tok in target])
#         all_predictions.append(predicted_text)
#         all_references.append(target_text)

#     results = rouge.compute(predictions=all_predictions, references=all_references)
#     return results


def evaluate_model(model, sample_texts, vocab, max_new_tokens=5, max_samples=None):
    device = next(model.parameters()).device
    idx2word = {v: k for k, v in vocab.items()}
    unk_id = vocab.get("<unk>", 1)
    eos_id = vocab.get("<eos>", 2)
    all_predictions = []
    all_references = []

    for i, text in enumerate(sample_texts):
        if max_samples is not None and i >= max_samples:
            break

        words = text.split()
        if len(words) < 4:
            continue

        cut = int(len(words) * 0.75)
        prefix_ids = [vocab.get(w, unk_id) for w in words[:cut]]
        
        target_ids = [vocab.get(w, unk_id) for w in words[cut:]] + [eos_id]
        target_words = [idx2word.get(tid, "<unk>") for tid in target_ids]

        if not prefix_ids or not target_words:
            continue

        input_ids = torch.tensor([prefix_ids]).to(device)
        lengths = torch.tensor([len(prefix_ids)])

        generated = model.generate(input_ids, lengths, max_new_tokens=max_new_tokens)
        new_tokens = generated[0][len(prefix_ids):]

        predicted_text = " ".join([idx2word.get(tok.item(), "<unk>") for tok in new_tokens])
        target_text = " ".join(target_words)

        all_predictions.append(predicted_text)
        all_references.append(target_text)

    return rouge.compute(predictions=all_predictions, references=all_references)