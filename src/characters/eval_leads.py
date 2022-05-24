import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    USE_COR_RES = True

    annotations_folder = Path(os.getcwd()) / 'data/aesop/annotations'
    res_folder = Path(os.getcwd()) / 'res/aesop/ner'
    leads_folder = Path(os.getcwd()) / 'res/aesop/leads/afinn'

    if USE_COR_RES:
        res_folder = f'{res_folder}/cor_res'

    stories = []
    for filename in os.listdir(annotations_folder):
        if filename.endswith('.json'):
            stories.append(filename)

    f1_scores = []
    precision_scores = []
    recall_scores = []

    preds = {}
    protagonists = []
    antagonists = []

    for story_name in stories:
        print(f'Processing story: "{story_name}"')
        with open(f'{annotations_folder}/{story_name}', 'r') as f:
            j_file = json.load(f)
            gt_prot = j_file['protagonist']
            gt_ant = j_file['antagonist']
            gt_chars = j_file['characters']

        with open(f'{res_folder}/{story_name}', 'r') as f:
            j_file = json.load(f)
            pred_chars = j_file['characters']

        with open(f'{leads_folder}/{story_name}', 'r') as f:
            j_file = json.load(f)
            pred_leads = j_file['leads']

        protagonists.append(gt_prot)
        antagonists.append(gt_ant)

        for key in pred_leads.keys():
            if key not in preds:
                preds[key] = {
                    'protagonists': [],
                    'antagonists': []
                }
            prot = pred_leads[key]['protagonist']
            ant = pred_leads[key]['antagonist']
            if prot is None:
                prot = ''
            if ant is None:
                ant = ''
            preds[key]['protagonists'].append(prot)
            preds[key]['antagonists'].append(ant)

    for key in preds.keys():
        print(f'Now displaying results for "{key}"')
        prot_acc = accuracy_score(protagonists, preds[key]['protagonists'])
        print(f'\tProtagonist accuracy: {prot_acc}')
        ant_acc = accuracy_score(antagonists, preds[key]['antagonists'])
        print(f'\tAntagonist accuracy: {ant_acc}')
