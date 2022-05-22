import os
from collections import Counter
from pathlib import Path

import stanza

from coreference_resolution import coreference_resolution
from utils import read_story

nlp = stanza.Pipeline('en', processors='tokenize,ner')


def NER(sentence, nlp):
    # perform ner
    doc = nlp(sentence)
    name_entity = [ent.text for ent in doc.ents if ent.type == 'PERSON']

    # convert all names to lowercase and remove 's in names
    name_entity = [str(x).lower().replace("'s", "") for x in name_entity]

    # remove article words
    name_entity = [x.split(' ') for x in name_entity]
    name_entity = [[word for word in x if not word in ['the', 'an', 'a', 'and']] for x in name_entity]
    name_entity = [' '.join(x) for x in name_entity]

    return name_entity


def name_entity_recognition(doc, use_cor_res=True):
    if use_cor_res:
        doc = coreference_resolution(doc)

    characters = NER(doc, nlp)
    counts = Counter(characters)
    characters = [x for x in counts]
    counts = [counts[x] for x in counts]

    return characters, counts, doc


if __name__ == '__main__':
    USE_COR_RES = True

    data_folder = Path(os.getcwd()) / 'data/aesop/original'

    name = 'The_Cock_and_the_Pearl'
    short_story = read_story(name, data_folder)

    characters, counts, doc = name_entity_recognition(short_story)
    print(characters)
    print(doc)
