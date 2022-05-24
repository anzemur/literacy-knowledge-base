# with helper functions from
# https://github.com/hzjken/character-network

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import stanza
from afinn import Afinn
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from name_entity_recognition import name_entity_recognition
from utils import read_story

data_folder = Path(os.getcwd()) / 'data/aesop/original'


sentiment_method = 'afinn'  # 'afinn', 'stanza'
if sentiment_method == 'stanza':
    sentiments_processor = stanza.Pipeline(lang='en', processors='tokenize,sentiment', tokenize_pretokenized=True)
else:
    sentiments_processor = Afinn()

target_dir_net = f'data/net/{sentiment_method}'
target_sentiment_dir = f'res/aesop/sentiments/{sentiment_method}'
target_leads_dir = f'res/aesop/leads/{sentiment_method}'
target_graphs_dir = f'res/aesop/graphs/{sentiment_method}'


def calculate_align_rate(sentence_list):
    '''
    Function to calculate the align_rate of the whole novel
    '''
    if sentiment_method == 'stanza':
        sentiment_score = []
        for sentence in sentence_list:
            doc = sentiments_processor(sentence)
            for doc_sentence in doc.sentences:
                sentiment_score.append(float(doc_sentence.sentiment))
    else:
        sentiment_score = [sentiments_processor.score(x) for x in sentence_list]

    align_rate = np.sum(sentiment_score) / len(np.nonzero(sentiment_score)[0]) * -2
    print(align_rate)

    return align_rate


def calculate_matrix(name_list, sentences, cor_res_sentences, align_rate):
    '''
    Function to calculate the co-occurrence matrix and sentiment matrix among all the top characters
    :param name_list: the list of names of the top characters in the novel.
    :param sentences: the list of sentences in the novel.
    :param align_rate: the sentiment alignment rate to align the sentiment score between characters due to the writing style of
    the author. Every co-occurrence will lead to an increase or decrease of one unit of align_rate.
    :return: the co-occurrence matrix and sentiment matrix.
    '''

    # calculate a sentiment score for each sentence in the novel
    # afinn = Afinn()
    # sentiment_score = [afinn.score(x) for x in sentences]

    if sentiment_method == 'stanza':
        sentiment_score = []
        for sentence in sentences:
            doc = sentiments_processor(sentence)
            for doc_sentence in doc.sentences:
                sentiment_score.append(float(doc_sentence.sentiment) - 1)
    else:
        sentiment_score = [sentiments_processor.score(x) for x in sentences]

    # replace name occurrences with names that can be vectorized
    for i in range(len(cor_res_sentences)):
        cor_res_sentences[i] = cor_res_sentences[i].lower()

        for name in name_list:
            tmp = name.split(" ")
            tmp = "_".join(tmp)
            cor_res_sentences[i] = cor_res_sentences[i].replace(name, tmp)

    for i in range(len(name_list)):
        tmp = name_list[i].split(" ")
        name_list[i] = "_".join(tmp)

    name_vec = CountVectorizer(vocabulary=name_list, binary=True)

    # calculate occurrence matrix and sentiment matrix among the top characters
    if (len(name_list) == 0):
        return np.array([]), np.array([]), np.array([]), np.array([])
    else:
        occurrence_each_sentence = name_vec.fit_transform(cor_res_sentences).toarray()

    co_occurrence_matrix = np.dot(occurrence_each_sentence.T, occurrence_each_sentence)
    sentiment_matrix = np.dot(occurrence_each_sentence.T, (occurrence_each_sentence.T * sentiment_score).T)
    sentiment_matrix += align_rate * co_occurrence_matrix
    co_occurrence_matrix = np.tril(co_occurrence_matrix)
    sentiment_matrix = np.tril(sentiment_matrix)

    # diagonals of the matrices are set to be 0 (co-occurrence of name itself is meaningless)
    shape = co_occurrence_matrix.shape[0]
    co_occurrence_matrix[[range(shape)], [range(shape)]] = 0
    sentiment_matrix[[range(shape)], [range(shape)]] = 0

    # get character sentiments
    character_sentiments = (np.sum(occurrence_each_sentence.T * sentiment_score, axis=1)).T
    character_occurences = (np.sum(occurrence_each_sentence.T, axis=1)).T

    # normalize
    divisor = np.abs(character_sentiments).max()
    if divisor == 0:
        divisor = 1
    character_sentiments = character_sentiments / divisor

    return co_occurrence_matrix, sentiment_matrix, character_sentiments, character_occurences


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
    if (shape > 1):
        normalized_matrix = matrix / np.max(np.abs(matrix))
    else:
        normalized_matrix = matrix

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
    if (len(name_list) > 1):
        normalized_frequency = np.array(name_frequency) / np.max(name_frequency)
    else:
        normalized_frequency = name_frequency

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

    # plt.savefig('res/graphs/' + plt_name + suffix + '.pdf')

    return G


def get_top_10_pagerank(G):
    N = G.number_of_nodes()
    if (N <= 1):
        return []
    try:
        pgrnk = nx.pagerank(G)
    except:
        return []
    pgrnk.update((key, value / (N - 1)) for key, value in pgrnk.items())

    sorted_pgrnk = sorted(pgrnk.items(), key=lambda item: item[1], reverse=True)[:10]
    return sorted_pgrnk


def get_pagerank_leads(G, character_sentiments, spaced_characters):
    top_10_pagerank = get_top_10_pagerank(G)

    protagonist = None
    antagonist = None
    for name, rank in top_10_pagerank:
        name_idx = spaced_characters.index(name)
        sentiment = rank * character_sentiments[name_idx]
        if protagonist is None and sentiment > 0:
            protagonist = name
        if antagonist is None and sentiment < 0:
            antagonist = name
        if protagonist is not None and antagonist is not None:
            break

    return protagonist, antagonist


def get_sentiment_leads(character_sentiments, spaced_characters):
    protagonist = None
    antagonist = None

    if (len(character_sentiments) > 1):
        protagonist_idx = np.argmax(character_sentiments)
        antagonist_idx = np.argmin(character_sentiments)

        if (protagonist_idx >= 0 and character_sentiments[protagonist_idx] > 0):
            protagonist = spaced_characters[protagonist_idx]
        if (antagonist_idx >= 0 and character_sentiments[antagonist_idx] < 0):
            antagonist = spaced_characters[antagonist_idx]

    return protagonist, antagonist


def get_occurence_leads(character_occurences, spaced_characters):
    protagonist = None
    antagonist = None

    if (len(character_occurences) > 1):
        protagonist_idx = np.argmax(character_occurences)
        if (protagonist_idx >= 0):
            character_occurences[protagonist_idx] = -1
            protagonist = spaced_characters[protagonist_idx]

        if (len(character_occurences) > 2):
            antagonist_idx = np.argmax(character_occurences)

            if (antagonist_idx >= 0 and protagonist_idx != antagonist_idx):
                character_occurences[antagonist_idx] = -1
                antagonist = spaced_characters[antagonist_idx]

    return protagonist, antagonist


def get_occurence_sentiment_leads(character_occurences, character_sentiments, spaced_characters):
    protagonist = None
    antagonist = None

    for _ in range(len(character_sentiments)):
        candidate_idx = np.argmax(character_occurences)
        if (candidate_idx >= 0):
            character_occurences[candidate_idx] = -1
            if (protagonist is None and character_sentiments[candidate_idx] > 0):
                protagonist = spaced_characters[candidate_idx]
            if (antagonist is None and character_sentiments[candidate_idx] < 0):
                antagonist = spaced_characters[candidate_idx]

        if protagonist is not None and antagonist is not None:
            break

    return protagonist, antagonist


def save_character_sentiments(name, sentiment_matrix, spaced_characters):
    divisor = 0
    if (len(spaced_characters) > 1):
        divisor = np.abs(sentiment_matrix).max()
    if (divisor == 0):
        divisor = 1

    sentiments = {}
    for i in range(len(spaced_characters)):
        sentiments[spaced_characters[i]] = {}
        for j in range(len(spaced_characters)):
            sentiments[spaced_characters[i]][spaced_characters[j]] = sentiment_matrix[i, j] / divisor

    true_name = name.split('.')[0]

    res_file = f'{target_sentiment_dir}/{true_name}.json'
    with open(res_file, 'w') as file:
        json.dump({
            'sentiments': sentiments
        }, file, indent=4)


def save_leads(name, lead_pairs):
    true_name = name.split('.')[0]

    obj = {}
    for (type, protagonist, antagonist) in lead_pairs:
        obj[type] = {
            'protagonist': protagonist,
            'antagonist': antagonist
        }

    res_file = f'{target_leads_dir}/{true_name}.json'
    with open(res_file, 'w') as file:
        json.dump({
            'leads': obj
        }, file, indent=4)


def character_sentiments(name, doc):
    characters, character_counts, cor_res_doc = name_entity_recognition(doc)

    sentences = sent_tokenize(doc)
    cor_res_sentences = sent_tokenize(cor_res_doc)
    align_rate = calculate_align_rate(sentences)

    co_occurrence_matrix, sentiment_matrix, character_sentiments, character_occurences = calculate_matrix(characters, sentences, cor_res_sentences, align_rate)

    spaced_characters = [' '.join(x.split('_')) for x in characters]

    # plot co-occurrence and sentiment graph
    plot_graph(spaced_characters, character_counts, co_occurrence_matrix, name, ' co-occurrence graph', 'co-occurrence')
    sentiment_graph = plot_graph(spaced_characters, character_counts, sentiment_matrix, name, ' sentiment graph', 'sentiment')
    plot_graph(spaced_characters, character_counts, sentiment_matrix, name, ' bare graph', 'bare')

    save_character_sentiments(name, sentiment_matrix + sentiment_matrix.T, spaced_characters)

    pr_protagonist, pr_antagonist = get_pagerank_leads(sentiment_graph, character_sentiments, spaced_characters)
    sent_protagonist, sent_antagonist = get_sentiment_leads(character_sentiments, spaced_characters)
    occur_protagonist, occur_antagonist = get_occurence_leads(character_occurences, spaced_characters)
    occur_sent_protagonist, occur_sent_antagonist = get_occurence_sentiment_leads(character_occurences, character_sentiments, spaced_characters)

    print(f'PageRank leads: protagonist = "{pr_protagonist}", antagonist = "{pr_antagonist}"')
    print(f'Sentiment leads: protagonist = "{sent_protagonist}", antagonist = "{sent_antagonist}"')
    print(f'Occurence leads: protagonist = "{occur_protagonist}", antagonist = "{occur_antagonist}"')
    print(f'Occurence sentiments leads: protagonist = "{occur_sent_protagonist}", antagonist = "{occur_sent_antagonist}"')

    save_leads(name, [
        ('pagerank', pr_protagonist, pr_antagonist),
        ('sentiment', sent_protagonist, sent_antagonist),
        ('occurences', occur_protagonist, occur_antagonist),
        ('occurences_sentiments', occur_sent_protagonist, occur_sent_antagonist)
    ])


if __name__ == '__main__':
    story_folder = Path(os.getcwd()) / 'data/aesop/original'

    stories = []
    for filename in os.listdir(story_folder):
        if filename.endswith('.txt'):
            stories.append(filename)

    for story_name in tqdm(stories):
        print(f'Processing story: "{story_name}"')
        short_story = read_story(story_name, story_folder)
        character_sentiments(story_name, short_story)
