# How to run the code

## Preparation
- run `pip install -r requirements.txt`

## Train Bert Model

- `cd Bert_NER`
- change the device in config.py to 'cpu' if you do not have a GPU
- `python main.py`

## Train Flair + LSTM-CRF Model

- `cd Flair`
- `python main.py`

## Train Iterated Dilated CNN Model

- See README.md file in dilated-cnn-ner folder