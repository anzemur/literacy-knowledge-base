# page rank, betweeness centrality, closeness centrality

import networkx as nx


def node_degree(G, report_f):
    N = G.number_of_nodes()
    nd = list(map(lambda x: (x[0], x[1] / N), G.degree()))

    report_f.write("\nNODE DEGREE\n")

    report_f.write("TOP 10:\n")
    sorted_nd = sorted(nd, key=lambda x: x[1], reverse=True)[:10]
    print(sorted_nd)

    for degree in sorted_nd:
        report_f.write("{} - {}({}): {}\n".format(degree[0], G.graph['node_labels'][int(degree[0]) - 1], G.graph['node_values'][int(degree[0]) - 1], degree[1]))


def clustering_coefficient(G, report_f):
    cc = nx.clustering(G)

    report_f.write("\nCLUSTERING COEFFICIENT\n")

    report_f.write("TOP 10:\n")
    sorted_cc = dict(sorted(cc.items(), key=lambda item: item[1], reverse=True)[:10])

    for key, value in sorted_cc.items():
        report_f.write("{} - {}({}): {}\n".format(key, G.graph['node_labels'][int(key) - 1], G.graph['node_values'][int(key) - 1], value))


def harmonic_mean_distance(G, report_f):
    N = G.number_of_nodes()
    hmd = nx.harmonic_centrality(G)
    hmd.update((key, value / (N - 1)) for key, value in hmd.items())

    report_f.write("\nHARMONIC MEAN DISTANCE\n")

    report_f.write("TOP 10:\n")
    sorted_hmd = dict(sorted(hmd.items(), key=lambda item: item[1], reverse=True)[:10])

    for key, value in sorted_hmd.items():
        report_f.write("{} - {}({}): {}\n".format(key, G.graph['node_labels'][int(key) - 1], G.graph['node_values'][int(key) - 1], value))


if __name__ == "__main__":
    report_f = open("report.txt", "w")
    node_labels = []
    node_values = []

    with open('./data/net/12_1ecbplus') as f:
        for line in (f.readlines()[4:]):
            if line[0] == '#':
                words = line.rstrip().split("\"")
                print(words)
                if len(words) == 3:
                    node_labels.append(str(words[1]))
                    node_values.append(str(words[2]))
            else:
                break

    G = nx.Graph(nx.read_edgelist('./data/net/20_1ecbplus'), node_labels=node_labels, node_values=node_values)

    node_degree(G, report_f)
    clustering_coefficient(G, report_f)
    harmonic_mean_distance(G, report_f)
