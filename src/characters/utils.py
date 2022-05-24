import codecs
import os
from collections import Counter


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def most_frequent(list):
    occurrence_count = Counter(list)
    return occurrence_count.most_common()


def read_story(story_name, path):
    book_list = os.listdir(path)
    book_list = [i for i in book_list if i.find(story_name) >= 0]
    novel = ''
    for i in book_list:
        with codecs.open(path / i, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read().replace('\r', ' ').replace('\n', ' ').replace("\'", "'")
        novel += ' ' + data

    return novel


def calculate_metrics(gt, pred):
    TP = len(set(gt).intersection(set(pred)))
    FP = len(pred) - TP
    FN = len(set(gt) - set(pred))

    return TP, FP, FN


def precision(TP, FP):
    if TP == 0 and FP == 0:
        return 1.0

    return TP / (TP + FP)


def recall(TP, FN):
    if TP == 0 and FN == 0:
        return 1.0

    return TP / (TP + FN)


def f1(precision, recall):
    if precision == 0 and recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
