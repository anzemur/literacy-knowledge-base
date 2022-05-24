import os
from collections import Counter
from pathlib import Path

import stanza
import spacy

from coreference_resolution import coreference_resolution
from utils import read_story

nlp_stanza = stanza.Pipeline('en', processors='tokenize,ner')
nlp_spacy = spacy.load('en_core_web_trf')


def NER(sentence, method):
    # perform ner
    if method == 'stanza':
        doc = nlp_stanza(sentence)
        name_entity = [ent.text for ent in doc.ents if ent.type == 'PERSON']
    else:
        doc = nlp_spacy(sentence)
        name_entity = [x for x in doc.ents if x.label_ in ['PERSON']]

    # convert all names to lowercase and remove 's in names
    name_entity = [str(x).lower().replace("'s", "") for x in name_entity]

    # remove article words
    name_entity = [x.split(' ') for x in name_entity]
    name_entity = [[word for word in x if not word in ['the', 'an', 'a', 'and']] for x in name_entity]
    name_entity = [' '.join(x) for x in name_entity]

    return name_entity


def name_entity_recognition(doc, use_cor_res=True, method='stanza'):
    if use_cor_res:
        doc = coreference_resolution(doc)

    characters = NER(doc, method)
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
