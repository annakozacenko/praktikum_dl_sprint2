# Text Autocomplete — LSTM vs DistilGPT-2

Проект по автодополнению текстов на базе датасета Sentiment140.  
Сравниваются: собственная LSTM-модель и предобученная distilgpt2.

## Задача

Модель получает начало текста (75%) и предсказывает продолжение (25%).  
Оценка качества — метрики ROUGE-1 и ROUGE-2.

## Структура проекта

praktikum_dl_sprint2/
├── data/ # Датасеты
├── src/
│ ├── data_utils.py # Загрузка, очистка, vocab
│ ├── next_token_dataset.py # torch.Dataset и collate_fn
│ ├── lstm_model.py # Класс LSTMModel
│ ├── lstm_train.py # Обучение LSTM
│ ├── eval_lstm.py # Оценка LSTM по ROUGE
│ └── eval_transformer_pipeline.py # Оценка distilgpt2
├── configs/config.py # Гиперпараметры
├── models/ # Веса модели, модель
├── solution.ipynb # Ноутбук с результатами и выводами
└── requirements.txt

## Результаты

Оценка на 500 примерах из val-выборки:

| Модель      | ROUGE-1 | ROUGE-2 | Размер  | Примечание        |
| ----------- | ------- | ------- | ------- | ----------------- |
| LSTM (val)  | 0.0745  | 0.0143  | 111 МБ  | Обучена на твитах |
| LSTM (test) | 0.0552  | 0.0058  | 111 МБ  | Финальная оценка  |
| DistilGPT-2 | 0.0560  | 0.0071  | ~350 МБ | Без дообучения    |

**LSTM превосходит distilgpt2** по обоим метрикам, будучи в 3× легче.  
Признаков переобучения не обнаружено (val/train gap < 0.1 на всех эпохах).  
Лучший val_loss = 2.1811 достигнут на эпохе 10 из 10.

## Архитектура LSTM

- Embedding(112 361, 128) → LSTM(128, num_layers=2, dropout=0.3) → Linear
- Параметров: 29 140 969 (~111 МБ)
- Словарь: 112 361 токенов, покрытие val/test ~55%

## Примеры предсказаний

| Префикс                    | LSTM             | DistilGPT-2                |
| -------------------------- | ---------------- | -------------------------- |
| "now i dont wanna face my" | "phone"          | "own life."                |
| "for once it"              | "is a good day"  | "was done."                |
| "follow if your an"        | "awesome person" | "cillary knowledge is not" |

## Гипотезы для улучшения

| #   | Гипотеза                           | Как проверить                                 | Ожидаемый эффект               |
| --- | ---------------------------------- | --------------------------------------------- | ------------------------------ |
| 1   | Увеличить `hidden_size` до 256–512 | Переобучить с новым конфигом                  | ROUGE ↑, модель тяжелее        |
| 2   | Fine-tune distilgpt2 на твитах     | HuggingFace `Trainer` + датасет твитов        | Оба ROUGE ↑↑                   |
| 3   | Увеличить число эпох до 20+        | Loss ещё не вышел на плато                    | ROUGE постепенно ↑             |
| 4   | Заменить word-токенизацию на BPE   | `tokenizers` (HuggingFace) + пересборка vocab | Покрытие ~55% → ~95%+, ROUGE ↑ |

## Запуск

```bash
pip install -r requirements.txt
python src/lstm_train.py         # обучение LSTM
jupyter notebook solution.ipynb  # результаты и выводы
```
