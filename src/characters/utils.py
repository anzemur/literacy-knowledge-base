import codecs
import os
from collections import Counter


def most_frequent(list):
    occurrence_count = Counter(list)
    return occurrence_count.most_common(1)[0][0]


def read_story(story_name, path):
    book_list = os.listdir(path)
    book_list = [i for i in book_list if i.find(story_name) >= 0]
    novel = ''
    for i in book_list:
        with codecs.open(path / i, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read().replace('\r', ' ').replace('\n', ' ').replace("\'", "'")
        novel += ' ' + data

    return novel
