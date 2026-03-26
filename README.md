# Text Autocomplete — LSTM vs DistilGPT-2

Проект по автодополнению текстов на базе датасета Sentiment140.  
Сравниваются: собственная LSTM-модель и предобученная distilgpt2.

## Задача

Модель получает начало текста (75%) и предсказывает продолжение (25%).  
Оценка качества — метрики ROUGE-1 и ROUGE-2.

## Структура проекта
text-autocomplete/
├── data/ # Датасеты (не в git, генерируются скриптом)
├── src/
│ ├── data_utils.py # Загрузка, очистка, vocab
│ ├── next_token_dataset.py # torch.Dataset и collate_fn
│ ├── lstm_model.py # Класс LSTMModel
│ ├── lstm_train.py # Обучение LSTM
│ ├── eval_lstm.py # Оценка LSTM по ROUGE
│ └── eval_transformer_pipeline.py # Оценка distilgpt2
├── configs/config.py # Гиперпараметры
├── models/ # Веса модели (не в git)
├── solution.ipynb # Ноутбук с результатами и выводами
└── requirements.txt


## Результаты

| Модель      | ROUGE-1 | ROUGE-2 |
|-------------|---------|---------|
| LSTM        | 0.ХXXX  | 0.XXXX  |
| DistilGPT-2 | 0.XXXX  | 0.XXXX  |

> Подставь реальные значения после обучения

## Примеры предсказаний

| Prefix | LSTM | DistilGPT-2 |
|--------|------|-------------|
| "just woke up and" | "feeling tired" | "feeling really good today" |

## Запуск

```bash
pip install -r requirements.txt
python src/lstm_train.py        # обучение LSTM
jupyter notebook solution.ipynb # результаты
```