import json
import os
from pathlib import Path

from utils import calculate_metrics, f1, precision, recall

if __name__ == '__main__':
    USE_COR_RES = True

    annotations_folder = Path(os.getcwd()) / 'data/aesop/annotations'
    res_folder = Path(os.getcwd()) / 'res/aesop/ner'

    if USE_COR_RES:
        res_folder = f'{res_folder}/cor_res'

    stories = []
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.json'):
            stories.append(filename)

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for story_name in stories:
        with open(f'{annotations_folder}/{story_name}', 'r') as f:
            gt = json.load(f)['characters']

        with open(f'{res_folder}/{story_name}', 'r') as f:
            pred = json.load(f)['characters']

        TP, FP, FN = calculate_metrics(gt, pred)
        p = precision(TP, FP)
        r = recall(TP, FN)

        f1_scores.append(f1(p, r))
        precision_scores.append(p)
        recall_scores.append(r)

        print(gt)
        print(pred)
        print('############')

    print(f'F1-score: {sum(f1_scores) / len(f1_scores)}')
    print(f'Precision: {sum(precision_scores) / len(precision_scores)}')
    print(f'Recall: {sum(recall_scores) / len(recall_scores)}')
