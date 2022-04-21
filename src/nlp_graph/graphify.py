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

# visulise the graph
import networkx as nx

nlp = spacy.load("en_core_web_lg")
vector_size = 300  # the size of token vector representation s in en_core_web_lg model

def text_to_graph(text, y):
    text_to_graph_start = time.time()

    edge_index = []  # list of all edges in a graph
    doc = nlp(text)
    root = 0
    for token in doc:
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
    return data

def text_to_nx(text):
    graph = text_to_graph(text, 0)

    graph_to_nx_start = time.time()
    g = pyg_utils.to_networkx(graph, to_undirected=True)
    graph_to_nx_end = time.time()

    print(f"Graph to nx took: {graph_to_nx_end - graph_to_nx_start:.2f} s")
    return g

def file_to_nx(filename):
    f = open(filename, "r")
    return text_to_nx(f.read())

if __name__ == '__main__':
    file_name = 'Leiningen_Vs_the_Ants'
    print(f'Now working on: {file_name}')
    G = file_to_nx(f'data/en/{file_name}.txt')
    print('No. nodes: ', G.number_of_nodes())
    print('No. edges: ', G.number_of_edges())

    giant = G.subgraph(max(nx.connected_components(G), key=len))
    print('No. nodes in LCSG: ', giant.number_of_nodes())
    print('No. edges in LCSG: ', giant.number_of_edges())

    
    nx_draw_start = time.time()
    f = plt.figure()
    nx.draw(G, ax=f.add_subplot(1, 1, 1))
    f.savefig(f"figures/{file_name}.pdf")
    plt.close(f)
    nx_draw_end = time.time()
    print(f"Full draw took: {nx_draw_end - nx_draw_start:.2f} s")

    nx_draw_start = time.time()
    f = plt.figure()
    nx.draw(giant, ax=f.add_subplot(1, 1, 1))
    f.savefig(f"figures/{file_name}_lcsg.pdf")
    plt.close(f)
    nx_draw_end = time.time()
    print(f"LCSG draw took: {nx_draw_end - nx_draw_start:.2f} s")