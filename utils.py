from __future__ import division

import pandas as pd
import numpy as np
from random import shuffle
import time
import sys

from collections import defaultdict, Counter, OrderedDict
from numpy import array, zeros, allclose

__author__ = 'gyennie'
DATA_FOLDER = "Data/Split_Data/"
DICT_FILE = "data/token_dict.csv"
EMBEDDINGS_FOLDER = "Data/"
LABEL_ENUM_STR = {'unrelated': 0, 'agree': 1, 'disagree': 2, 'discuss': 3}
LABEL_ENUM_INT = {0: 'unrelated', 1: 'agree', 2: 'disagree', 3: 'discuss'}


def convert_to_text(body):
    token_frame = pd.read_csv(DICT_FILE, index_col=0, header=None)
    text_body = []
    for word in body:
        text_body.append(token_frame.ix[word].values[0])

    return text_body


def load_bodies(data_set='Train'):
    """
    Load body data
    :param data_set: Train, Dev, Test
    :return: list of numpy arrays for each body's text (tokenized to integers) and a list of body ids (in order)
                or None if error
    """
    if data_set == "Train":
        filename = 'train_bodies.csv'
    elif data_set == "Dev":
        filename = 'dev_bodies.csv'
    elif data_set == "Test":
        filename = 'test_bodies.csv'
    else:
        return None

    body_frame = pd.read_csv(DATA_FOLDER+filename, index_col=0, header=None)
    bodies = []
    body_ids = []
    for index, row in body_frame.iterrows():
        body_ids.append(index)
        str_list = str(row.values).strip("[").strip("]").strip("'").strip("\"").strip().split(",")
        bodies.append(np.array([int(x.strip('\"').strip("\'")) for x in str_list]))
    del body_frame
    return bodies, body_ids


def load_stances(data_set="Train"):
    """
        Load stance data
        :param data_set: Train, Dev, Test
        :return: list of a list of numpy arrays for each stance (tokenized to integers) and a list of body ids
                (in order) and a list of a list of labels
                    or None if error
                structure:
                ids[i] - body id i
                stances[i][j] - body id i, stance j
                labels[i][j] - body id i, stance j

        :example:
            stances, labels, ids = load_stances("Test")
            print ids[0], stances[0][0], labels[0][0] -  all should correlate
        """
    if data_set == "Train":
        filename = 'train_stances.csv'
    elif data_set == "Dev":
        filename = 'dev_stances.csv'
    elif data_set == "Test":
        filename = 'test_stances.csv'
    else:
        return None

    stance_frame = pd.read_csv(DATA_FOLDER + filename, index_col=0, header=None)
    stances = []
    body_ids = []
    labels = []
    for index, row in stance_frame.iterrows():
        stance_stance = []
        label_label = []
        body_ids.append(index)
        stance_list = str(row.values).split(";")
        for each in stance_list:
            sup = each.strip("\'").strip("\"").strip("[").strip("]").strip("\'").split(",")
            label_label.append(sup[-1].strip("\"").strip().strip("]").strip("[").strip("\'"))
            buff_arr = []
            for each_each in sup[:-1]:
                el = each_each.strip("\"").strip("\'").strip().strip("]").strip("[").strip("\"").strip("\'").strip()
                buff_arr.append(int(el.strip("[").strip("\'")))
            stance_stance.append(np.array(buff_arr))
        stances.append(stance_stance)

        labels.append([LABEL_ENUM_STR[l] for l in label_label])
    del stance_frame
    return stances, labels, body_ids


def get_minibatches(batch_size, max_body_len, max_stance_len, randomize=True, data_set='Train'):
    bodies, body_body_ids = load_bodies(data_set)
    stances, labels, stance_body_ids = load_stances(data_set)

    unpacked_bodies = []
    unpacked_stances = []
    unpacked_labels = []
    indices = []
    index = 0
    for i in range(len(stances)):
        for j in range(len(stances[i])):
            indices.append(index)
            index += 1
            unpacked_bodies.append(bodies[i][:max_body_len])
            unpacked_stances.append(stances[i][j][:max_stance_len])
            unpacked_labels.append(labels[i][j])

    if randomize:
        shuffle(indices)
    batches = []
    while len(indices) > 0:
        batch_bodies = []
        batch_stances = []
        batch_labels = []
        batch_body_length = []
        batch_stance_length = []
        for i in range(batch_size):
            if len(indices) < 1:
                break
            cur_index = indices.pop(0)
            batch_body_length.append(len(unpacked_bodies[cur_index]))
            batch_stance_length.append(len(unpacked_stances[cur_index]))
            batch_bodies.append(unpacked_bodies[cur_index].tolist() + [0 for i in range(max_body_len - batch_body_length[-1])])
            batch_stances.append(unpacked_stances[cur_index].tolist() + [0 for i in range(max_stance_len - batch_stance_length[-1])])
            batch_labels.append(unpacked_labels[cur_index])
        if len(batch_labels) == batch_size:
            batches.append({'bodies': np.array(batch_bodies),
                            'stances': np.array(batch_stances),
                            'labels': np.array(batch_labels),
                            'body_len': np.array(batch_body_length),
                            'stance_len': np.array(batch_stance_length)})

    return batches


def get_debug_training_set(batch_size, max_body_len, max_stance_len):
    batches = get_minibatches(batch_size, max_body_len, max_stance_len, False, 'Train')
    return [batches[0], batches[1], batches[2]]


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)

def to_table(data, row_labels, column_labels, precision=2, digits=4):
    """Pretty print tables.
    Assumes @data is a 2D array and uses @row_labels and @column_labels
    to display table.
    """
    # Convert data to strings
    data = [["%04.2f"%v for v in row] for row in data]
    cell_width = max(
        max(map(len, row_labels)),
        max(map(len, column_labels)),
        max(max(map(len, row)) for row in data))
    def c(s):
        """adjust cell output"""
        return s + " " * (cell_width - len(s))
    ret = ""
    ret += "\t".join(map(c, column_labels)) + "\n"
    for l, row in zip(row_labels, data):
        ret += "\t".join(map(c, [l] + row)) + "\n"
    return ret

class ConfusionMatrix(object):
    """
    A confusion matrix stores counts of (true, guessed) labels, used to
    compute several evaluation metrics like accuracy, precision, recall
    and F1.
    """

    def __init__(self, labels, default_label=None):
        self.labels = labels
        self.default_label = default_label if default_label is not None else len(labels) -1
        self.counts = defaultdict(Counter)

    def update(self, gold, guess):
        """Update counts"""
        self.counts[gold][guess] += 1

    def as_table(self):
        """Print tables"""
        # Header
        data = [[self.counts[l][l_] for l_,_ in enumerate(self.labels)] for l,_ in enumerate(self.labels)]
        return to_table(data, self.labels, ["go\\gu"] + self.labels)

    def summary(self, quiet=False):
        """Summarize counts"""
        keys = range(len(self.labels))
        data = []
        macro = array([0., 0., 0., 0.])
        micro = array([0., 0., 0., 0.])
        default = array([0., 0., 0., 0.])
        for l in keys:
            tp = self.counts[l][l]
            fp = sum(self.counts[l_][l] for l_ in keys if l_ != l)
            tn = sum(self.counts[l_][l__] for l_ in keys if l_ != l for l__ in keys if l__ != l)
            fn = sum(self.counts[l][l_] for l_ in keys if l_ != l)

            acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
            prec = (tp)/(tp + fp) if tp > 0  else 0
            rec = (tp)/(tp + fn) if tp > 0  else 0
            f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0

            # update micro/macro averages
            micro += array([tp, fp, tn, fn])
            macro += array([acc, prec, rec, f1])
            if l != self.default_label: # Count count for everything that is not the default label!
                default += array([tp, fp, tn, fn])

            data.append([acc, prec, rec, f1])

        # micro average
        tp, fp, tn, fn = micro
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])
        # Macro average
        data.append(macro / len(keys))

        # default average
        tp, fp, tn, fn = default
        acc = (tp + tn)/(tp + tn + fp + fn) if tp > 0  else 0
        prec = (tp)/(tp + fp) if tp > 0  else 0
        rec = (tp)/(tp + fn) if tp > 0  else 0
        f1 = 2 * prec * rec / (prec + rec) if tp > 0  else 0
        data.append([acc, prec, rec, f1])

        # Macro and micro average.
        return to_table(data, self.labels + ["micro","macro","not-O"], ["label", "acc", "prec", "rec", "f1"])
