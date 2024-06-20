import torch
import torch.nn as nn
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans


#调整节点编号，使节点从0开始计数
def adjust_node_labels(G):
    mapping = {node: node - 1 for node in G.nodes}
    H = nx.relabel_nodes(G, mapping)
    return H

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
    # G = copy.deepcopy(G0)
    input = torch.ones(len(G), 11)
    for i in G.nodes():
        # print(i)
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




# 核心点采样
def core_point_sampling(node_features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(node_features)
    clusters = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    core_indices = []
    for i in range(n_clusters):
        cluster_indices = np.where(clusters == i)[0]
        cluster_features = node_features[cluster_indices]
        center = cluster_centers[i].reshape(1, -1)
        closest, _ = pairwise_distances_argmin_min(center, cluster_features)
        core_point = cluster_indices[closest[0]]
        core_indices.append(core_point)

    return core_indices


# 不确定性采样
def uncertainty_sampling(predictions_list, core_indices, n_select):
    core_predictions = np.array([predictions_list[i] for i in core_indices])
    # 计算不确定性，可以使用预测分数的方差
    uncertainty = np.var(core_predictions, axis=1)  # 计算每个核心点的预测分数的方差
    selected_indices = np.argsort(uncertainty)[-n_select:]  # 选择方差最大的几个点

    return [core_indices[i] for i in selected_indices]

def read_labels_from_file(file_path):
    labels = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            node_id = int(parts[0])
            label = float(parts[1])
            labels[node_id] = label
    return labels

def rmse(value,train_nodes, labels):
    loss = nn.MSELoss()
    labels = torch.tensor(labels).reshape(-1, 1)
    y = torch.cat([value[node] for node in train_nodes]).reshape(-1,1)
    nodelabels = torch.cat([labels[node] for node in train_nodes]).reshape(-1,1)
    rmse = torch.sqrt(loss(y,nodelabels))
    return rmse.item()

def train(train_nodes, nodes_list, model, node_labels, g, node_features,G):
    data_train = [x for x in nodes_list if x in train_nodes]  # 训练集
    data_test = [x for x in nodes_list if x not in train_nodes]  # 测试
    # # print(model)
    print("train_data:", data_train)
    print("test_data", data_test)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 优化器
    loss = nn.MSELoss()
    train_ls, test_ls, train_logls, test_logls = [], [], [], []  # 训练损失随训练次数曲线
    # 微调前测试集loss
    model.eval()
    value = model(g, node_features)
    test_loss = rmse(value, [x for x in data_test], node_labels)
    test_ls.append(test_loss)
    model.train()

    # 开始微调
    for epoch in range(150):
        print("epoch:", epoch)
        nodes = [x for x in train_nodes]
        nodelabels = [node_labels[node].item() for node in train_nodes]
        value = model(g, node_features)
        y = torch.cat([value[node].unsqueeze(1) for node in train_nodes], 0)
        train_labels = torch.tensor(nodelabels).reshape(-1, 1)
        y = torch.clamp(y, min=1e-8)
        l = loss(torch.log(y), torch.log(train_labels))
        optimizer.zero_grad()
        l.backward()
        # 使用梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        value = model(g, node_features)
        train_ls.append(rmse(value, nodes, node_labels))
        model.eval()
        value = model(g, node_features)
        test_ls.append(rmse(value, [x for x in data_test], node_labels))
        model.train()
    plt.figure()
    plt.title('%s'%len(G.nodes()))
    plt.plot(list(range(len(test_ls))),test_ls)
    plt.show()
    plt.close()
    model.eval()
    with torch.no_grad():  # 在推理时不需要计算梯度
        predictions = model(g, node_features)
    predictions = predictions.tolist()
    preresult = []
    for i in range(len(predictions)):
        item = (i, predictions[i][0])
        preresult.append(item)

    return preresult

def GNNTAL(G):
    # 生成标签
    # generatelabel(G)
    labels_file_path = 'SIRLabels/Dolphins.txt'
    labels_dict = read_labels_from_file(labels_file_path)
    print(labels_dict)
    y = torch.tensor([labels_dict[i] for i in range(len(G.nodes()))], dtype=torch.float)

    g = dgl.from_networkx(G)
    node_features_ = get_dgl_g_input(G)
    node_features = torch.cat((node_features_[:, 0:8], node_features_[:, 9:11]), dim=1)
    nodes_list = list(G.nodes())

    model = torch.load('GNNT.pth')

    # 提取核心点
    n_clusters = 20  # 设置聚类数目
    core_indices = core_point_sampling(node_features, n_clusters)
    print(core_indices)

    model.eval()
    with torch.no_grad():  # 在推理时不需要计算梯度
        predictions = model(g, node_features)
    predictions = predictions.tolist()
    print("first predictions:", predictions)

    # 不确定性采样选择节点
    n_select = int(0.1*len(G.nodes()))  # 设置选择的节点数目
    print("n_select", n_select)
    selected_core_point_indices = uncertainty_sampling(predictions, core_indices, n_select)

    print("Selected core points with highest uncertainty:", selected_core_point_indices)

    model.train()
    train_nodes = selected_core_point_indices
    preresult = train(train_nodes, nodes_list, model, y, g, node_features, G)
    print(preresult)
    return preresult
