from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from typing import List


if __name__ == "__main__":
  # 1. get the corpus
  columns = {0: 'text', 1: 'ner'}
  data_folder = 'data/partial'
  corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file='train',
                                test_file='dev.out')
      
  tag_type = 'ner'

  tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

  # initialize embeddings
  embedding_types: List[TokenEmbeddings] = [

      # GloVe embeddings
      WordEmbeddings('glove'),

      # contextual string embeddings, forward
      PooledFlairEmbeddings('news-forward', pooling='min'),

      # contextual string embeddings, backward
      PooledFlairEmbeddings('news-backward', pooling='min')
  ]

  embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)


  tagger: SequenceTagger = SequenceTagger(hidden_size=512,
                                          embeddings=embeddings,
                                          tag_dictionary=tag_dictionary,
                                          tag_type=tag_type,
                                          use_crf=True,
                                          use_rnn=True)

  # Train
  trainer: ModelTrainer = ModelTrainer(tagger, corpus)

  trainer.train(f'flair_model',
                learning_rate=0.05,
                mini_batch_size=64,
                train_with_dev=True,              
                max_epochs=150)

  # Test
  columns = {0: 'text'}
  data_folder = 'test/partial'
  corpus: Corpus = ColumnCorpus(data_folder, columns,
                                train_file='test.in',
                                test_file='test.in')

  model = SequenceTagger.load('flair_model/final-model.pt')
  model.evaluate(corpus.test, mini_batch_size=16, num_workers=4, out_path="test.p6.model.out", embedding_storage_mode="none")

  def read_data(path, column=0):
      """column=0 means input sequence, column=1 means label
      """
      with open(path) as f:
          lines = f.readlines()
      
      data = []
      sample = []
      
      for line in lines:
          formatted_line = line.strip()
          
          if len(formatted_line) > 0:
              split_data = formatted_line.split(" ")
              sample.append(split_data[column])

          else:
              data.append(sample)
              sample = []
              
      return data


  text, y_preds = read_data("test.p6.model.out", 0), read_data("test.p6.model.out", 2)
  lines = []
  for words, tags in zip(text, y_preds):
      for w, t in zip(words, tags):
          lines.append(f"{w} {t}\n")
      lines.append("\n")

  with open("test.p6.model.out", "w", encoding="utf-8") as outfile:
      outfile.write("".join(lines))
