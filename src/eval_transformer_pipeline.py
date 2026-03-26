from transformers import pipeline
import evaluate

rouge = evaluate.load("rouge")
generator = pipeline("text-generation", model="distilgpt2")


# def evaluate_transformer(dataset, max_new_tokens=5, max_samples=None):
#     all_predictions = []
#     all_references = []

#     for text in dataset:
#         if len(all_predictions) >= max_samples:
#             break

#         words = text.split()
#         if len(words) < 4:
#             continue

#         n = len(words)
#         cut = int(n * 0.75)
#         prefix = " ".join(words[:cut])
#         target = " ".join(words[cut:])

#         result = generator(prefix, max_new_tokens=max_new_tokens, do_sample=True, top_k = 50 )
#         generated_text = result[0]["generated_text"]

#         new_text = generated_text[len(prefix) :].strip()
#         all_predictions.append(new_text)
#         all_references.append(target)

#     results = rouge.compute(predictions=all_predictions, references=all_references)

#     return results



def evaluate_transformer(sample_texts, max_new_tokens=5, max_samples=None):
    all_predictions = []
    all_references = []

    for i, text in enumerate(sample_texts):
        if max_samples is not None and i >= max_samples:
            break

        words = text.split()
        if len(words) < 4:
            continue

        cut = int(len(words) * 0.75)
        prefix = " ".join(words[:cut])
        target = " ".join(words[cut:])

        result = generator(prefix, max_new_tokens=max_new_tokens, do_sample=True, top_k=50)
        new_text = result[0]["generated_text"][len(prefix):].strip()

        all_predictions.append(new_text)
        all_references.append(target)

    return rouge.compute(predictions=all_predictions, references=all_references)