from datetime import datetime
import spacy
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

# torch geometric libraries
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.graphgym import optim
import torch_geometric.utils as pyg_utils
import time
import os

# visulise the graph
import networkx as nx

source_dir = 'data/en'
target_dir_net = 'data/net'
target_dir_figures = 'figures'

nlp = spacy.load("en_core_web_lg")
vector_size = 300  # the size of token vector representation s in en_core_web_lg model

def text_to_graph(text, y):
    text_to_graph_start = time.time()

    node_labels = {}
    edge_index = []  # list of all edges in a graph
    doc = nlp(text)
    root = 0
    for token in doc:
        node_labels[token.i] = token.text
        edge_index.append([token.i, token.i])  # add a self loop
        if token.i == token.head.i:
            root = token.i
        else:
            edge_index.append([token.i, token.head.i])  # add a connection from token to its parent
            edge_index.append([token.head.i, token.i])  # add a reverse connection
    
    x = torch.tensor(np.array([d.vector for d in doc]))  # compute token embedings
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # torch geometric expects to get the edges in a matrix with to rows and a column for each connection
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([y]), text=text, root_index=root)
    
    text_to_graph_end = time.time()
    print(f"Text to graph took: {text_to_graph_end - text_to_graph_start:.2f} s")
    return data, node_labels

def text_to_nx(text):
    graph, labels = text_to_graph(text, 0)

    graph_to_nx_start = time.time()
    g = pyg_utils.to_networkx(graph, to_undirected=True)
    nx.set_node_attributes(g, labels, 'label')
    graph_to_nx_end = time.time()

    print(f"Graph to nx took: {graph_to_nx_end - graph_to_nx_start:.2f} s")
    return g, labels

def file_to_nx(filename):
    f = open(filename, "r")
    return text_to_nx(f.read())

def process_file(filename):
    print(f'Now working on: {filename}')
    G, labels = file_to_nx(f'{source_dir}/{filename}.txt')
    print('No. nodes: ', G.number_of_nodes())
    print('No. edges: ', G.number_of_edges())
    nx.write_gexf(G, f'{target_dir_net}/{filename}.gexf')

    giant = G.subgraph(max(nx.connected_components(G), key=len))
    print('No. nodes in LCSG: ', giant.number_of_nodes())
    print('No. edges in LCSG: ', giant.number_of_edges())
    nx.write_gexf(giant, f'{target_dir_net}/{filename}_lcsg.gexf')

    
    nx_draw_start = time.time()
    f = plt.figure()
    nx.draw(nx.relabel_nodes(G, labels), ax=f.add_subplot(1, 1, 1), with_labels = True)
    f.savefig(f"{target_dir_figures}/{filename}.pdf")
    plt.close(f)
    nx_draw_end = time.time()
    print(f"Full draw took: {nx_draw_end - nx_draw_start:.2f} s")

    nx_draw_start = time.time()
    f = plt.figure()
    nx.draw(nx.relabel_nodes(giant, labels), ax=f.add_subplot(1, 1, 1), with_labels = True)
    f.savefig(f"{target_dir_figures}/{filename}_lcsg.pdf")
    plt.close(f)
    nx_draw_end = time.time()
    print(f"LCSG draw took: {nx_draw_end - nx_draw_start:.2f} s")

if __name__ == '__main__':
    for file in os.listdir(source_dir):
        if file.endswith(".txt"):
            split = file.split('.')
            filename = split[0]
            print('=================')
            process_file(filename)