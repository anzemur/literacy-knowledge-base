import xml.etree.ElementTree as ET
import itertools
import codecs, os, spacy, json
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from afinn import Afinn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer

file_source = 'characterR/test.xml'
tree = ET.parse(file_source)
root = tree.getroot()
print(f'processing {file_source}')

tokens = root.findall('token')
short_story = ""
for x in tokens:
    short_story += x.text + " "
#print(short_story)

#people
entities = []
for x in root.findall('Markables')[0].findall('HUMAN_PART_PER'):
    #print(x)
    id_text = ''
    if len(x) == 0:
        #print('\t>>> no tokens, skipped')
        continue
    for x_token in x:
        for token in tokens:
            if token.attrib['t_id'] == x_token.attrib['t_id']:
                id_text += token.text.lower()
                break
    #entities.append((x.attrib['m_id'], 'PER', id_text))
    entities.append(id_text)

#preprocessing
name_entities = [str(x).lower().replace("'s","") for x in entities]
name_entities = list(set(name_entities)) # remove duplicates

def top_names(name_list, novel, top_num=20):
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
    # to calculate the align_rate of the whole novel
    afinn = Afinn()
    sentiment_score = [afinn.score(x) for x in sentence_list]
    align_rate = np.sum(sentiment_score)/len(np.nonzero(sentiment_score)[0]) * -2

    return align_rate

def calculate_matrix(name_list, sentence_list, align_rate):
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
    for i in lower_tri_loc:
        edge_list.append((name_list[i[0]], name_list[i[1]], {'weight': weight[i], 'color': color[i]}))

    return edge_list

def plot_graph(name_list, name_frequency, matrix, plt_name, mode, path=''):
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

    if mode == 'co-occurrence':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 4000, edge_cmap=plt.cm.Blues,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True, width=weights)
    elif mode == 'sentiment':
        nx.draw(G, pos, node_color='#A0CBE2', node_size=np.sqrt(normalized_frequency) * 4000,
                linewidths=10, font_size=35, labels=label, edge_color=colors, with_labels=True,
                width=weights, edge_vmin=-1000, edge_vmax=1000)
    else:
        raise ValueError("mode should be either 'co-occurrence' or 'sentiment'")

    plt.savefig('characterR/graphs/' + plt_name + '.png')

#main
sentence_list = sent_tokenize(short_story)
align_rate = calculate_align_rate(sentence_list)
name_frequency, name_list = top_names(name_entities, short_story, 20)
cooccurrence_matrix, sentiment_matrix = calculate_matrix(name_list, sentence_list, align_rate)
# plot co-occurrence and sentiment graph
plot_graph(name_list, name_frequency, cooccurrence_matrix, "xml" + ' co-occurrence graph', 'co-occurrence')
plot_graph(name_list, name_frequency, sentiment_matrix, "xml" + ' sentiment graph', 'sentiment')
