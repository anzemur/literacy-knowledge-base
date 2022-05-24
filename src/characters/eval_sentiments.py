import json
import os
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support


if __name__ == '__main__':
    USE_COR_RES = True

    annotations_folder = Path(os.getcwd()) / 'data/aesop/annotations'
    char_folder = Path(os.getcwd()) / 'res/aesop/ner'
    sent_folder = Path(os.getcwd()) / 'res/aesop/sentiments/afinn'

    if USE_COR_RES:
        char_folder = f'{char_folder}/cor_res'

    stories = []
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.json'):
            stories.append(filename)

    f1_scores = []
    precision_scores = []
    recall_scores = []

    for story_name in stories:
        print(f'Processing story: "{story_name}"')
        with open(f'{annotations_folder}/{story_name}', 'r') as f:
            j_file = json.load(f)
            gt_sents = j_file['sentiments']
            gt_chars = j_file['characters']

        with open(f'{char_folder}/{story_name}', 'r') as f:
            j_file = json.load(f)
            pred_chars = j_file['characters']

        with open(f'{sent_folder}/{story_name}', 'r') as f:
            j_file = json.load(f)
            pred_sents = j_file['sentiments']

        if (len(pred_chars) != len(set(pred_chars).intersection(set(gt_chars)))):
            continue

        gt_sentiments = []
        pred_sentiments = []

        for key in gt_sents.keys():
            if key not in pred_sents:
                continue
            for subkey in gt_sents[key].keys():
                if subkey not in pred_sents:
                    continue
                gt_sentiments.append(gt_sents[key][subkey])
                pred_sentiments.append(pred_sents[key][subkey])

        pred_sentiments_clean = []  # we need rounding for score computation
        for pred_sent in pred_sentiments:
            if pred_sent < -0.33:
                pred_sentiments_clean.append(-1)
            elif pred_sent < 0.33:
                pred_sentiments_clean.append(0)
            else:
                pred_sentiments_clean.append(1)

        p, r, f1, _ = precision_recall_fscore_support(gt_sentiments, pred_sentiments_clean, labels=[-1, 0, 1], average='micro')

        f1_scores.append(f1)
        precision_scores.append(p)
        recall_scores.append(r)

        print('scores:', p, r, f1)

        print(gt_sents)
        print(pred_sents)
        print('############')

    print(f'Eligible files: {len(f1_scores)}')
    print(f'F1-score: {sum(f1_scores) / len(f1_scores)}')
    print(f'Precision: {sum(precision_scores) / len(precision_scores)}')
    print(f'Recall: {sum(recall_scores) / len(recall_scores)}')
