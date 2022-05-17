#with helper functions from
#https://github.com/hzjken/character-network

import codecs
import os
import spacy
import json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from afinn import Afinn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from  nltk import word_tokenize, pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

target_dir_net="data/net"

def common_words(path):
    '''
    A function to read-in the top common words from external .txt document.
    :param path: The path where the common words info is stored.
    :return: A set of the top common words.
    '''

    with codecs.open(path) as f:
        words = f.read()
        words = json.loads(words)

    return set(words)

def flatten(input_list):
    '''
    A function to flatten complex list.
    :param input_list: The list to be flatten
    :return: the flattened list.
    '''

    flat_list = []
    for i in input_list:
        if type(i) == list:
            flat_list += flatten(i)
        else:
            flat_list += [i]

    return flat_list

def read_novel(story_name, path):
    '''
    A function to read-in the novel text from given path.
    :param book_name: The name of the novel.
    :param path: The path where the novel text file is stored.
    :return: the novel text.
    '''

    book_list = os.listdir(path)
    book_list = [i for i in book_list if i.find(story_name) >= 0]
    novel = ''
    for i in book_list:
        with codecs.open(path / i, 'r', encoding='utf-8', errors='ignore') as f:
            data = f.read().replace('\r', ' ').replace('\n', ' ').replace("\'", "'")
        novel += ' ' + data

    return novel

def named_entities_from_senctence(sentence, model='nlp'):
    if model == 'nlp':
        doc = nlp(sentence)
        return [x for x in doc.ents if x.label_ in ['PERSON']]
    elif model == 'nltk':
        entities = []
        chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
        for chunk in chunks:
            if (hasattr(chunk, 'label') and chunk.label() == 'PERSON'):
                person = ' '.join(c[0] for c in chunk)
                entities.append(person)
        return entities
    else:
        print('Unknown model')
        exit(1)

def get_entities(sentence):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  #############################################################
  
  for tok in nlp(sentence):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
      
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      ## chunk 3
      if tok.dep_.find("subj") == True:
        # ent1 = modifier +" "+ prefix + " "+ tok.text
        ent1 = prefix + " "+ tok.text
        if (tok.tag_ == 'PRP'):
            # print('entity 1', ent1, tok.tag_, list(tok.ancestors), list(tok.children))
            # ent1 = tok.text
            ent1 = ''
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""      

      ## chunk 4
      if tok.dep_.find("obj") == True:
        # ent2 = modifier +" "+ prefix +" "+ tok.text
        ent2 = prefix +" "+ tok.text
        if (tok.tag_ == 'PRP'):
            # print('entity 2', ent2, tok.tag_, list(tok.ancestors), list(tok.children))
            # ent2 = tok.text
            ent2 = ''
        
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return [ent1.strip(), ent2.strip()]

def name_entity_recognition(sentence):
    '''
    A function to retrieve name entities in a sentence.
    :param sentence: the sentence to retrieve names from.
    :return: a name entity list of the sentence.
    '''

    # doc = nlp(sentence)
    # # retrieve person from the sentence
    # name_entity = [x for x in doc.ents if x.label_ in ['PERSON']]
    entity_pair = get_entities(sentence=sentence)
    name_entity = named_entities_from_senctence(sentence=sentence, model='nltk')
    # convert all names to lowercase and remove 's in names
    name_entity = [str(x).lower().replace("'s","") for x in name_entity]
    # split names into single words ('Harry Potter' -> ['Harry', 'Potter'])
    name_entity = [x.split(' ') for x in name_entity]
    # flatten the name list
    name_entity = flatten(name_entity)
    # remove name words that are less than 3 letters to raise recognition accuracy
    name_entity = [x for x in name_entity if len(x) >= 3]
    # remove name words that are in the set of 4000 common words
    name_entity = [x for x in name_entity if x not in words]

    return (name_entity, entity_pair)

def iterative_NER(sentence_list, threshold_rate=0.0005):
    '''
    A function to execute the name entity recognition function iteratively. The purpose of this
    function is to recognise all the important names while reducing recognition errors.
    :param sentence_list: the list of sentences from the novel
    :param threshold_rate: the per sentence frequency threshold, if a word's frequency is lower than this
    threshold, it would be removed from the list because there might be recognition errors.
    :return: a non-duplicate list of names in the novel.
    '''
    entity_pairs = []
    output = []
    for i in sentence_list:
        name_list, entity_pair = name_entity_recognition(i)
        if name_list != []:
            output.append(name_list)
        
        entity_pairs.append(entity_pair)

    output = flatten(output)
    from collections import Counter
    output = Counter(output)
    output = [x for x in output if output[x] >= threshold_rate * len(sentence_list)]

    entities = flatten(entity_pairs)
    entities = filter(lambda p: len(p) > 0, entities)
    entities = Counter(entities)
    entities = [x for x in entities if entities[x] > 1]

    # return output
    return entities


def top_names(name_list, novel, top_num=20):
    '''
    A function to return the top names in a novel and their frequencies.
    :param name_list: the non-duplicate list of names of a novel.
    :param novel: the novel text.
    :param top_num: the number of names the function finally output.
    :return: the list of top names and the list of top names' frequency.
    '''

    vect = CountVectorizer(vocabulary=name_list, stop_words='english')
    name_frequency = vect.fit_transform([novel.lower()])
    name_frequency = pd.DataFrame(name_frequency.toarray(), columns=vect.get_feature_names_out())
    name_frequency = name_frequency.T
    name_frequency = name_frequency.sort_values(by=0, ascending=False)
    name_frequency = name_frequency[0:top_num]
    names = list(name_frequency.index)
    name_frequency = list(name_frequency[0])

    return name_frequency, names


def calculate_align_rate(sentence_list):
    '''
    Function to calculate the align_rate of the whole novel
    :param sentence_list: the list of sentence of the whole novel.
    :return: the align rate of the novel.
    '''
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    align_rate = np.sum(sentiment_score)/len(np.nonzero(sentiment_score)[0]) * -2

    return align_rate


def calculate_matrix(name_list, sentence_list, align_rate):
    '''
    Function to calculate the co-occurrence matrix and sentiment matrix among all the top characters
    :param name_list: the list of names of the top characters in the novel.
    :param sentence_list: the list of sentences in the novel.
    :param align_rate: the sentiment alignment rate to align the sentiment score between characters due to the writing style of
    the author. Every co-occurrence will lead to an increase or decrease of one unit of align_rate.
    :return: the co-occurrence matrix and sentiment matrix.
    '''

    # calculate a sentiment score for each sentence in the novel
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    # calculate occurrence matrix and sentiment matrix among the top characters
    name_vect = CountVectorizer(vocabulary=name_list, binary=True)
    occurrence_each_sentence = name_vect.fit_transform(sentence_list).toarray()
    cooccurrence_matrix = np.dot(occurrence_each_sentence.T, occurrence_each_sentence)
    sentiment_matrix = np.dot(occurrence_each_sentence.T, (occurrence_each_sentence.T * sentiment_score).T)
    sentiment_matrix += align_rate * cooccurrence_matrix
    cooccurrence_matrix = np.tril(cooccurrence_matrix)
    sentiment_matrix = np.tril(sentiment_matrix)
    # diagonals of the matrices are set to be 0 (co-occurrence of name itself is meaningless)
    shape = cooccurrence_matrix.shape[0]
    cooccurrence_matrix[[range(shape)], [range(shape)]] = 0
    sentiment_matrix[[range(shape)], [range(shape)]] = 0

    return cooccurrence_matrix, sentiment_matrix


def matrix_to_edge_list(matrix, mode, name_list):
    '''
    Function to convert matrix (co-occurrence/sentiment) to edge list of the network graph. It determines the
    weight and color of the edges in the network graph.
    :param matrix: co-occurrence matrix or sentiment matrix.
    :param mode: 'co-occurrence' or 'sentiment'
    :param name_list: the list of names of the top characters in the novel.
    :return: the edge list with weight and color param.
    '''
    edge_list = []
    shape = matrix.shape[0]
    lower_tri_loc = list(zip(*np.where(np.triu(np.ones([shape, shape])) == 0)))
    normalized_matrix = matrix / np.max(np.abs(matrix))
    if mode == 'co-occurrence':
        weight = np.log(2000 * normalized_matrix + 1) * 0.7
        color = np.log(2000 * normalized_matrix + 1)
    if mode == 'sentiment':
        weight = np.log(np.abs(1000 * normalized_matrix) + 1) * 0.7
        color = 2000 * normalized_matrix
    if mode == 'bare':
        weight = np.log(np.abs(1000 * normalized_matrix) + 1) * 0.7
        color = 2000 * normalized_matrix
    for i in lower_tri_loc:
        # print('edge weight', weight[i])
        if (mode != 'bare' or weight[i] > 0.0001):
            edge_list.append((name_list[i[0]], name_list[i[1]], {'weight': weight[i], 'color': color[i]}))

    return edge_list


def plot_graph(name_list, name_frequency, matrix, plt_name, suffix, mode, path=''):
    '''
    Function to plot the network graph (co-occurrence network or sentiment network).
    :param name_list: the list of top character names in the novel.
    :param name_frequency: the list containing the frequencies of the top names.
    :param matrix: co-occurrence matrix or sentiment matrix.
    :param plt_name: the name of the plot (PNG file) to output.
    :param mode: 'co-occurrence' or 'sentiment'
    :param path: the path to output the PNG file.
    :return: a PNG file of the network graph.
    '''

    label = {i: i for i in name_list}
    edge_list = matrix_to_edge_list(matrix, mode, name_list)
    normalized_frequency = np.array(name_frequency) / np.max(name_frequency)

    plt.figure(figsize=(20, 20))
    G = nx.Graph()
    G.add_nodes_from(name_list)
    G.add_edges_from(edge_list)
    pos = nx.circular_layout(G)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    colors = [G[u][v]['color'] for u, v in edges]

    if mode == 'bare':
        nx.write_gexf(G, f'{target_dir_net}/{plt_name}_characters.gexf')
    elif mode == 'sentiment':
        nx.write_gexf(G, f'{target_dir_net}/{plt_name}_character_sentiment.gexf')

    if mode == 'co-occurrence':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 4000, edge_cmap=plt.cm.Blues,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True, width=weights)
    elif mode == 'sentiment':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 4000,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True,
                width=weights, edge_vmin=-1000, edge_vmax=1000)
    elif mode == 'bare':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 4000,
                linewidths=10, font_size=35, labels=label, with_labels=True, edge_vmin=-1000, edge_vmax=1000)
    else:
        raise ValueError("mode should be either 'bare', 'co-occurrence', or 'sentiment'")

    plt.savefig('characterR/graphs/' + plt_name + suffix + '.png')


if __name__ == '__main__':
    # nlp = spacy.load('en_core_web_sm')
    nlp = spacy.load('en_core_web_lg')
    words = common_words('characterR/common_words.txt')
    data_folder = Path(os.getcwd()) / 'data/en/grimm'

    # try one story
    name = 'LITTLE_RED_CAP'

    short_story = read_novel(name, data_folder)

    sentence_list = sent_tokenize(short_story)
    align_rate = calculate_align_rate(sentence_list)
    preliminary_name_list = iterative_NER(sentence_list)
    name_frequency, name_list = top_names(preliminary_name_list, short_story, 30)
    cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list, align_rate)
    # plot co-occurrence and sentiment graph
    plot_graph(name_list, name_frequency, cooccurrence_matrix, name, ' co-occurrence graph', 'co-occurrence')
    plot_graph(name_list, name_frequency, sentiment_matrix, name, ' sentiment graph', 'sentiment')
    plot_graph(name_list, name_frequency, sentiment_matrix, name, ' bare graph', 'bare')

    '''
    # loop over all stories
    short_stories = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".txt"):
            short_stories.append(filename)

    for name in short_stories:
        short_story = read_novel(name, data_folder)
        sentence_list = sent_tokenize(short_story)
        align_rate = calculate_align_rate(sentence_list)
        preliminary_name_list = iterative_NER(sentence_list)
        name_frequency, name_list = top_names(preliminary_name_list, short_story, 20)
        cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list, align_rate)
        # plot co-occurrence and sentiment graph
        plot_graph(name_list, name_frequency, cooccurrence_matrix, name + ' co-occurrence graph', 'co-occurrence')
        plot_graph(name_list, name_frequency, sentiment_matrix, name + ' sentiment graph', 'sentiment')
    '''
