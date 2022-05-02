# page rank, betweeness centrality, closeness centrality

import networkx as nx
import os

source_dir = 'data/net'
neighbour_range = 10

def node_degree(G, report_f):
    N = G.number_of_nodes()
    nd = list(map(lambda x: (x[0], x[1] / N), G.degree()))

    report_f.write("\nNODE DEGREE\n")

    report_f.write("TOP 10:\n")
    sorted_nd = sorted(nd, key=lambda x: x[1], reverse=True)[:10]

    nodes = G.nodes
    for key, value in sorted_nd:
        if key.isnumeric():
            targets = []
            for i in range(int(key) - neighbour_range, int(key) + neighbour_range + 1):
                if str(i) in nodes:
                    
                    if str(i) == key:
                        targets.append(f"|{nodes[str(i)]['label']}|")
                    else:
                        targets.append(nodes[str(i)]['label'])
            label = ' '.join(targets)
        else:
            label = nodes[key]['label']

        report_f.write("{} - {}: {}\n".format(key, label, value))


def clustering_coefficient(G, report_f):
    cc = nx.clustering(G)

    report_f.write("\nCLUSTERING COEFFICIENT\n")

    report_f.write("TOP 10:\n")
    sorted_cc = dict(sorted(cc.items(), key=lambda item: item[1], reverse=True)[:10])

    nodes = G.nodes
    for key, value in sorted_cc.items():
        if key.isnumeric():
            targets = []
            for i in range(int(key) - neighbour_range, int(key) + neighbour_range + 1):
                if str(i) in nodes:
                    
                    if str(i) == key:
                        targets.append(f"|{nodes[str(i)]['label']}|")
                    else:
                        targets.append(nodes[str(i)]['label'])
            label = ' '.join(targets)
        else:
            label = nodes[key]['label']
        report_f.write("{} - {}: {}\n".format(key, label, value))


def harmonic_mean_distance(G, report_f):
    N = G.number_of_nodes()
    hmd = nx.harmonic_centrality(G)
    hmd.update((key, value / (N - 1)) for key, value in hmd.items())

    report_f.write("\nHARMONIC MEAN DISTANCE\n")

    report_f.write("TOP 10:\n")
    sorted_hmd = dict(sorted(hmd.items(), key=lambda item: item[1], reverse=True)[:10])

    nodes = G.nodes
    for key, value in sorted_hmd.items():
        if key.isnumeric():
            targets = []
            for i in range(int(key) - neighbour_range, int(key) + neighbour_range + 1):
                if str(i) in nodes:
                    
                    if str(i) == key:
                        targets.append(f"|{nodes[str(i)]['label']}|")
                    else:
                        targets.append(nodes[str(i)]['label'])
            label = ' '.join(targets)
        else:
            label = nodes[key]['label']
        report_f.write("{} - {}: {}\n".format(key, label, value))



def pagerank(G, report_f):
    N = G.number_of_nodes()
    pgrnk = nx.pagerank(G)
    pgrnk.update((key, value / (N - 1)) for key, value in pgrnk.items())

    report_f.write("\nPAGERANK\n")

    report_f.write("TOP 10:\n")
    sorted_pgrnk = dict(sorted(pgrnk.items(), key=lambda item: item[1], reverse=True)[:10])

    nodes = G.nodes
    for key, value in sorted_pgrnk.items():
        if key.isnumeric():
            targets = []
            for i in range(int(key) - neighbour_range, int(key) + neighbour_range + 1):
                if str(i) in nodes:
                    
                    if str(i) == key:
                        targets.append(f"|{nodes[str(i)]['label']}|")
                    else:
                        targets.append(nodes[str(i)]['label'])
            label = ' '.join(targets)
        else:
            label = nodes[key]['label']
        report_f.write("{} - {}: {}\n".format(key, label, value))


if __name__ == "__main__":
    report_f = open("report.txt", "w")

    for file in os.listdir(source_dir):
        if file.endswith(".gexf"):
            split = file.split('.')
            filename = split[0]
            G = nx.Graph(nx.read_gexf(f'./{source_dir}/{filename}.gexf'))
            report_f.write(f'==========================\n')
            report_f.write(f'Processing: {filename}\n')
            node_degree(G, report_f)
            # clustering_coefficient(G, report_f)
            harmonic_mean_distance(G, report_f)
            pagerank(G, report_f)

    report_f.close()
