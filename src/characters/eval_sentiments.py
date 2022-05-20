import json
import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support



if __name__ == '__main__':
    USE_COR_RES = True

    annotations_folder = Path(os.getcwd()) / 'data/aesop/annotations'
    res_folder = Path(os.getcwd()) / 'res/aesop/sentiments'

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
            gt = json.load(f)['sentiments']

        with open(f'{res_folder}/{story_name}', 'r') as f:
            pred = json.load(f)['sentiments']

        if (len(pred.keys()) != len(set(pred.keys()).intersection(set(gt.keys())))):
            continue

        gt_sentiments = []
        pred_sentiments = []

        for key in gt.keys():
            if key not in pred:
                continue
            for subkey in gt[key].keys():
                if subkey not in pred:
                    continue
                gt_sentiments.append(gt[key][subkey])
                pred_sentiments.append(pred[key][subkey])

        pred_sentiments_clean = [] # we need rounding for score computation

        for pred in pred_sentiments:
            if pred < -0.33:
                pred_sentiments_clean.append(-1)
            elif pred < 0.33:
                pred_sentiments_clean.append(0)
            else:
                pred_sentiments_clean.append(1)

        p, r, f1, _ = precision_recall_fscore_support(gt_sentiments, pred_sentiments)

        f1_scores.append(f1)
        precision_scores.append(p)
        recall_scores.append(r)

        print(gt)
        print(pred)
        print('############')

    print(f'F1-score: {sum(f1_scores) / len(f1_scores)}')
    print(f'Precision: {sum(precision_scores) / len(precision_scores)}')
    print(f'Recall: {sum(recall_scores) / len(recall_scores)}')
