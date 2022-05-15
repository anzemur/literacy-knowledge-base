import networkx as nx
import os

import community
import collections
import matplotlib.pyplot as plt

source_dir = 'data/net'

if __name__ == "__main__":
    # for file in os.listdir(source_dir):
    #     if file.endswith(".gexf"):
            file = 'The_Most_Dangerous_Game_characters.gexf'
            split = file.split('.')
            filename = split[0]
            G = nx.Graph(nx.read_gexf(f'./{source_dir}/{filename}.gexf'))

            partition = community.best_partition(G)
            values = [partition.get(node) for node in G.nodes()]
            counter=collections.Counter(values)
            # sp = nx.spring_layout(G)
            sp = nx.circular_layout(G)
            nx.draw_networkx(G, pos=sp, with_labels=True, node_size=50, node_color=values)
            # plt.axes('off')
            plt.show()