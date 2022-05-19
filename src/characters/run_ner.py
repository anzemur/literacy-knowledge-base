import json
import os
from pathlib import Path

from name_entity_recognition import name_entity_recognition
from utils import read_story

if __name__ == '__main__':
    USE_COR_RES = True

    data_folder = Path(os.getcwd()) / 'data/aesop/original'
    res_folder = Path(os.getcwd()) / 'res/aesop/ner'

    if USE_COR_RES:
        res_folder = f'{res_folder}/cor_res'

    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    print(res_folder)

    stories = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            stories.append(filename.split(".")[0])

    for story_name in stories:
        story = read_story(story_name, data_folder)
        characters, _, _ = name_entity_recognition(story, USE_COR_RES)

        res_file = f'{res_folder}/{story_name}.json'
        with open(res_file, 'w') as file:
            json.dump({'characters': characters}, file, indent=4)
