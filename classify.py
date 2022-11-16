import math
from copy import deepcopy
from typing import Dict, Any
import os  # learned functions from https://docs.python.org/3/library/os.html

def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    vocab = []
    words = {}

    sub_dir_list = [os.path.join(directory, dir_n) for dir_n in os.listdir(directory)]
    for sub_dir in sub_dir_list:
        for file in os.listdir(sub_dir):
            try:
                with open(os.path.join(sub_dir, file), "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for word in lines:
                        word = word.strip()
                        if word in words:
                            words.update({word: words[word] + 1})
                        else:
                            words.update({word: 1})
            except FileExistsError and FileNotFoundError:
                print(f"ERROR: File not found (PATH: {file}).")
    for word in words:
        if words[word] >= cutoff:
            vocab.append(word)
    vocab.sort()
    return vocab


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for word in lines:
            word = word.strip()
            if word in vocab and word in bow:
                bow.update({word: bow[word] + 1})
            elif word in vocab and word not in bow:
                bow.update({word: 1})
            elif None in bow:
                bow.update({None: bow[None] + 1})
            else:
                bow.update({None: 1})
    return bow


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []

    sub_dir_list = [os.path.join(directory, dir_n) for dir_n in os.listdir(directory)]
    for sub_dir in sub_dir_list:
        sub_name = sub_dir.split('/')[-1]
        for file in os.listdir(sub_dir):
            bow = create_bow(vocab, os.path.join(sub_dir, file))
            dataset.append({"label": sub_name, "bow": bow})
    return dataset


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob: Dict[Any, Any] = {}

    n_files = len(training_data)
    for label in label_list:
        n_labels = [data["label"] for data in training_data].count(label)
        prob = float(math.log(n_labels + smooth) - math.log(n_files + len(label_list)))
        logprob.update({label: prob})
    return logprob


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """

    smooth = 1  # smoothing factor
    word_prob = {}

    for word in vocab:
        word_prob.update({word: word_probability(word, label, training_data, smooth, vocab)})
    word_prob.update({None: word_probability(None, label, training_data, smooth, vocab)})
    return word_prob


def word_probability(word, label, training_data, smooth, vocab):
    word_p = 0
    label_words = 0
    for data in training_data:
        if label == data["label"]:
            for w in data["bow"]:
                if w == word:
                    word_p += data["bow"][w]
                label_words += data["bow"][w]
    word_p = float(math.log(word_p + smooth * 1) - math.log(label_words + smooth * (len(vocab) + 1)))
    return word_p


##################################################################################
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}

    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    retval.update({"vocabulary": vocab})
    retval.update({"log prior": prior(training_data, ['2020', '2016'])})
    retval.update({"log p(w|y=2020)": p_word_given_label(vocab, training_data, '2020')})
    retval.update({"log p(w|y=2016)": p_word_given_label(vocab, training_data, '2016')})
    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}

    wordp_2016 = float()
    wordp_2020 = float()
    bow = create_bow(model["vocabulary"], filepath)
    for word in bow:
        if word in model["log p(w|y=2016)"]:
            for x in range(bow[word]):
                wordp_2016 += model["log p(w|y=2016)"][word]
        if word in model["log p(w|y=2020)"]:
            for x in range(bow[word]):
                wordp_2020 += model["log p(w|y=2020)"][word]
    p2016 = float(model["log prior"]["2016"] + wordp_2016)
    p2020 = float(model["log prior"]["2020"] + wordp_2020)

    if p2016 > p2020:
        prediction = '2016'
    else:
        prediction = '2020'

    retval.update({"predicted y": prediction})
    retval.update({"log p(y=2016|x)": p2016})
    retval.update({"log p(y=2020|x)": p2020})
    return retval

