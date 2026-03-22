from transformers import pipeline
import evaluate

rouge = evaluate.load("rouge")
generator = pipeline("text-generation", model="distilgpt2")


def evaluate_transformer(dataset, max_new_tokens=5, max_samples=200):
    all_predictions = []
    all_references = []

    for text in dataset:
        if len(all_predictions) >= max_samples:
            break

        words = text.split()
        if len(words) < 4:
            continue

        n = len(words)
        cut = int(n * 0.75)
        prefix = " ".join(words[:cut])
        target = " ".join(words[cut:])

        result = generator(prefix, max_new_tokens=max_new_tokens, do_sample=False)
        generated_text = result[0]["generated_text"]

        new_text = generated_text[len(prefix) :].strip()
        all_predictions.append(new_text)
        all_references.append(target)

    results = rouge.compute(predictions=all_predictions, references=all_references)

    return results
