# dilated-cnn-ner

This code implements the models described in the paper
"[Fast and Accurate Entity Recognition with Iterated Dilated Convolutions](https://arxiv.org/abs/1702.02098)"
by [Emma Strubell](https://cs.umass.edu/~strubell), [Patrick Verga](https://cs.umass.edu/~pat),
[David Belanger](https://cs.umass.edu/~belanger) and [Andrew McCallum](https://cs.umass.edu/~mccallum).

Requirements
-----
This code uses TensorFlow v[1.0, 1.4) and Python 2.7.

Setup
-----
1. Set up environment variables. For example, from the root directory of this project:

  ```
  export DILATED_CNN_NER_ROOT=`pwd`
  export DATA_DIR=/path/to/conll-2003
  ```

2. Get the glove word embedding 

   ```
   cd data/embeddings
   wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
   unzip glove.6B.zip
   ```

3. Perform all data preprocessing:

  ```
  cd ../..
  ./bin/preprocess.sh conf/projectdata/dilated-cnn.conf
  ```

  This calls `preprocess.py`, which loads the data from text files, maps the tokens, labels and any other features to
  integers, and writes to TensorFlow tfrecords.

Training
----
Once the data preprocessing is completed, you can train a tagger:

  ```
  ./bin/train-cnn.sh conf/projectdata/dilated-cnn.conf
  ```
Evaluation
----
Once the training is completed, it will generate an output file `pred.txt` on test set specified in config file. You can evaluate the output by:

  ```
  mv pred.txt eval
  cd eval
  python collneval.py < pred.txt
  ```

Configs
----
Configuration files (`conf/*`) specify all the data, parameters, etc. for an experiment.
