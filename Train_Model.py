import networkx as nx
import pandas as pd
import dgl
import torch
import torch.nn as nn
import warnings
from Model import GNNT
warnings.filterwarnings('ignore')
def get_neigbors(g, node, depth):
    output = {}
    layers = dict(nx.bfs_successors(g, source=node, depth_limit=depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output

def get_dgl_g_input(G):
    input = torch.ones(len(G), 11)
    for i in G.nodes():
        input[i, 0] = G.degree()[i]
        input[i, 1] = sum([G.degree()[j] for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)
        input[i, 2] = sum([nx.clustering(G, j) for j in list(G.neighbors(i))]) / max(len(list(G.neighbors(i))), 1)
        egonet = G.subgraph(list(G.neighbors(i)) + [i])
        input[i, 3] = len(egonet.edges())
        input[i, 4] = sum([G.degree()[j] for j in egonet.nodes()]) - 2 * input[i, 3]

    for l in [1, 2, 3]:
        for i in G.nodes():
            ball = get_neigbors(G, i, l)
            input[i, 5 + l - 1] = (G.degree()[i] - 1) * sum([G.degree()[j] - 1 for j in ball[l]])

    v = nx.voterank(G)
    votescore = dict()
    for i in list(G.nodes()): votescore[i] = 0
    for i in range(len(v)):
        votescore[v[i]] = len(G) - i
    e = nx.eigenvector_centrality(G, max_iter=1000)
    k = nx.core_number(G)
    for i in G.nodes():
        input[i, 8] = votescore[i]
        input[i, 9] = e[i]
        input[i, 10] = k[i]
    for i in range(len(input[0])):
        if max(input[:, i]) != 0:
            input[:, i] = input[:, i] / max(input[:, i])
    return input

if __name__=='__main__':
    sage_para = dict([["in_dim", 10], ["out_dim", 32], ["embed_dim", 32], ["activation", "elu"]])
    transformer_para = dict([["d_model", 32],["nhead", 4],["num_encoder_layers",2],["num_decoder_layers",2],["dim_feedforward",32],["dropout",0.1]])
    gnnt = GNNT(sage_para, transformer_para)
    Edge = pd.read_csv('D://DataSet/.csv')  # Network Data
    u = list(Edge['u'])
    v = list(Edge['v'])
    edge_list = [(u[i], v[i]) for i in range(len(v))]
    G = nx.Graph()
    G.add_edges_from(edge_list)
    node_features_ = get_dgl_g_input(G)
    print(gnnt)
    print(node_features_.shape)
    g = dgl.from_networkx(G)
    node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)
    output = gnnt(g,node_features)
    G = nx.convert_node_labels_to_integers(G)
    data = pd.read_csv("..\\dataset\\synthetic\\train_1000_4_Influence.csv")  # Label
    data_memory = [list(data.loc[i]) for i in range(len(data))]
    print("data_memory", data_memory)
    for x in data_memory: x[0] = int(x[0])
    g = dgl.from_networkx(G)
    nodes_list = list(G.nodes())
    num_epochs = 500
    lr = 0.001
    model = gnnt
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    train_logls = []
    train_ls = []
    nodes_list = [x[0] for x in data_memory]
    train_nodes = nodes_list
    data_train = [x for x in data_memory if x[0] in train_nodes]
    for epoch in range(num_epochs):
        nodes = [x[0] for x in data_train]
        labels = [x[1] for x in data_train]
        value = model(g,node_features)
        y = torch.cat([value[node].unsqueeze(1) for node in nodes], 0)
        train_labels = torch.tensor(labels).reshape(-1,1)
        l=loss(torch.log(y),torch.log(train_labels))
        if l!=l:
            model.sage1.reset_parameters()
            model.sage2.reset_parameters()
            model.fc1.reset_parameters()
            model.fc2.reset_parameters()
            continue
        train_ls.append(l.detach().numpy())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        print('epoch:%s, ' % epoch, 'train_ls:%s, ' % train_ls[-1])
    torch.save(model,'GNNT.pth')