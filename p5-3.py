import argparse
import codecs
from collections import defaultdict, Counter
import json
import re
import numpy as np
import sys
import random
from conlleval import evaluate

np.set_printoptions(precision=4)


def read_conll_file(file_name):
    """
    read in a file with format:
    word1    tag1
    ...      ...
    wordN    tagN
    Sentences MUST be separated by newlines!
    :param file_name: file to read in
    :return: generator of instances ((list of  words, list of tags) pairs)
    """
    current_words = []
    current_tags = []
    current_pos = []

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            word, pos, tag = line.split()
            current_words.append(word)
            current_tags.append(tag)
            current_pos.append(pos)

        else:
            yield (current_words, current_tags, current_pos)
            current_words = []
            current_tags = []
            current_pos = []

    # if file does not end in newline (it should...), check whether there is an instance in the buffer
    if current_tags != []:
        yield (current_words, current_tags, current_pos)

def memoize(f):
    """
    helper function to be used as decorator to memoize features
    :param f:
    :return:
    """
    memo = {}
    def helper(*args):
        key = tuple(args[1:])
        try:
            return memo[key]
        except KeyError:
            memo[key] = f(*args)
            return memo[key]
    return helper



class StructuredPerceptron(object):
    """
    implements a structured perceptron as described in Collins 2002
    """

    def __init__(self, seed=1512141834):
        """
        initialize model
        :return:
        """
        self.feature_weights = defaultdict(float)
        self.tags = set()

        self.START = "__START__"
        self.END = "__END__"
        random.seed(seed)
        np.random.seed(seed)
        

    def fit(self, train_data, iterations=5, learning_rate=0.2):
        averaged_weights = Counter()

        for iteration in range(iterations):
            correct = 0
            total = 0.0
            for i, (words, tags, poss) in enumerate(train_data):
                for tag in tags:
                    self.tags.add(tag)

                # get prediction
                prediction = self.decode(words, poss)

                # derive global features
                global_gold_features = self.get_global_features(words, tags, poss)
                global_prediction_features = self.get_global_features(words, prediction, poss)

                # update weight vector
                for fid, count in global_gold_features.items():
                    self.feature_weights[fid] += learning_rate * count
                for fid, count in global_prediction_features.items():
                    self.feature_weights[fid] -= learning_rate * count

                # compute training accuracy for this iteration
                correct += sum([1 for (predicted, gold) in zip(prediction, tags) if predicted == gold])
                total += len(tags)

            averaged_weights.update(self.feature_weights)
            print('\tTraining accuracy: %.4f' % (correct/total))

            random.shuffle(train_data)

        self.feature_weights = averaged_weights


    def get_global_features(self, words, tags, poss):
        """
        count how often each feature fired for the whole sentence
        :param words:
        :param tags:
        :return:
        """
        feature_counts = Counter()

        for i, (word, tag, pos) in enumerate(zip(words, tags, poss)):
            previous_tag = self.START if i == 0 else tags[i-1]
            feature_counts.update(self.get_features(word, tag, previous_tag, pos))

        return feature_counts


    @memoize
    def get_features(self, word, tag, previous_tag, pos):
        """
        get all features that can be derived from the word and tags
        :param word:
        :param tag:
        :param previous_tag:
        :return:
        """
        # word_lower = word.lower()
        # prefix = word_lower[:3]
        # suffix = word_lower[-3:]

        features = [
                    # 'TAG_%s' % (tag),                       # current tag
                    'TAG_BIGRAM_%s_%s' % (previous_tag, tag),  # tag bigrams
                    'WORD+TAG_%s_%s' % (word, tag),            # word-tag combination
                    'POS+TAG_%s_%s' % (pos, tag),
                    'COMBINE_%s_%s_%s' % (previous_tag, tag, word),

                    # 'WORD_LOWER+TAG_%s_%s' % (word_lower, tag),# word-tag combination (lowercase)
                    # 'UPPER_%s_%s' % (word[0].isupper(), tag),  # word starts with uppercase letter
                    # 'DASH_%s_%s' % ('-' in word, tag),         # word contains a dash
                    # 'PREFIX+TAG_%s_%s' % (prefix, tag),        # prefix and tag
                    # 'SUFFIX+TAG_%s_%s' % (suffix, tag),        # suffix and tag

                    #########################
                    # ADD MOAAAAR FEATURES! #
                    #########################
                    # ('WORDSHAPE', self.shape(word), tag),
                    # 'WORD+TAG_BIGRAM_%s_%s_%s' % (word, tag, previous_tag),
                    # 'SUFFIX+2TAGS_%s_%s_%s' % (suffix, previous_tag, tag),
                    # 'PREFIX+2TAGS_%s_%s_%s' % (prefix, previous_tag, tag)
        ]
        return features

    @memoize
    def shape(self, word):
        """
        some simple shape features
        """
        result = []
        for c in word:
            if c.isupper():
                result.append('X')
            elif c.islower():
                result.append('x')
            elif c in '0123456789':
                result.append('d')
            else:
                result.append(c)
        # replace multiple occurrences of a character with 'x*' and return it
        return re.sub(r"x+", "x*", ''.join(result))

    def decode(self, words, poss):
        """
        Find best sequence
        :param words:
        :return:
        """
        N=len(words)
        M=len(self.tags) #number of tags
        tags=list(self.tags)

        # create trellis of size M (number of tags) x N (sentence length)
        Q = np.ones((len(self.tags), N)) * float('-Inf')
        backp = np.ones((len(self.tags), N), dtype=np.int16) * -1 #backpointers

        ### initialization step
        cur_word=words[0]
        for j in range(M):
            # initialize probs for tags j at position 1 (first word)
            cur_tag=tags[j]
            features = self.get_features(words[0], cur_tag, self.START, poss[0])
            feature_weights = sum((self.feature_weights[x] for x in features))
            Q[j,0]=feature_weights

        # iteration step
        # filling the lattice, for every position and every tag find viterbi score Q
        for i in range(1,N):
            # for every tag
            for j in range(M):
                # checks if we are at end or start
                tag=tags[j]

                best_score = float('-Inf')

                # for every possible previous tag
                for k in range(M):

                    # k=previous tag
                    previous_tag=tags[k]

                    best_before=Q[k,i-1] # score until best step before

                    features = self.get_features(words[i], tag, previous_tag, poss[i])
                    feature_weights = sum((self.feature_weights[x] for x in features))

                    score = best_before + feature_weights

                    if score > best_score:
                        Q[j,i]=score
                        best_score = score
                        backp[j,i]=k #best tag

        # final best
        #best_id=np.argmax(Q[:, -1]) #the same
        best_id=Q[:,-1].argmax()

        ## print best tags in reverse order
        predtags=[]
        predtags.append(tags[best_id])

        for i in range(N-1,0,-1):
            idx=int(backp[best_id,i])
            predtags.append(tags[idx])
            best_id=idx

        #return reversed predtags
        #return (words,predtags[::-1])
        return predtags[::-1]


    def predict_eval(self, test_data, output=False):
        """
        compute accuracy on a test file
        :param file_name:
        :param output:
        :return:
        """
        f = open('full/dev.p5.SP.out', 'w')

        for i, (words, tags, poss) in enumerate(test_data):

            # get prediction
            prediction = self.decode(words, poss)

            if output:
                for word, gold, pred in zip(words, tags, prediction):
                    f.write(("{} {} {}\n".format(word, gold, pred)))
                f.write('\n')


        f.close()


    def save(self, file_name):
        """
        save model
        :param file_name:
        :return:
        """
        print("saving model...", end=' ', file=sys.stderr)
        with codecs.open(file_name, "w", encoding='utf-8') as model:
            model.write("%s\n" % json.dumps({'tags': list(self.tags), 'weights': dict(self.feature_weights)}))
        print("done", file=sys.stderr)


    def load(self, file_name):
        """
        load model from JSON file
        :param file_name:
        :return:
        """
        print("loading model...", end=' ', file=sys.stderr)
        model_data = codecs.open(file_name, 'r', encoding='utf-8').readline().strip()
        model = json.loads(model_data)
        self.tags = set(model['tags'])
        self.feature_weights = model['weights']
        print("done", file=sys.stderr)


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


if __name__=="__main__":
    train = 'full/train'
    test = "full/dev.out"
    output = True
    load = False
    save = "save/p5-3.json"
    iterations = 10
    learning_rate = 0.2
    

    # create new model
    sp = StructuredPerceptron()

    if load:
        sp.load(load)
        print("Model loaded successfully")
    else:
        train_data = list(read_conll_file(train))
        sp.fit(train_data, iterations=iterations, learning_rate=learning_rate)

        sp.save(save)
        print(f"Model weights saved to {save}")

    print("Inference...")
    # check whether to show predictions
    test_data = list(read_conll_file(test))
    sp.predict_eval(test_data, output=output)
    print("Done")
    print('*******\n')


    print("Evaluating")
    y_label = read_data('full/dev.p5.SP.out', 1)
    y_preds = read_data('full/dev.p5.SP.out', 2)
    
    y_label = [oo for o in y_label for oo in o+['O']]
    y_preds = [oo for o in y_preds for oo in o+['O']]
    print('Evaluation on SP')
    prec, rec, f1 = evaluate(y_label, y_preds, verbose=False)
    print(f'precision: {prec:.3f} \t rec: {rec:.3f} \t f1 {f1:.3f}')
