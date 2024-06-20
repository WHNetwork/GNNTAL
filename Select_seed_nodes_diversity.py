def select_seed_nodes_diversity(G, spreadability_list, k):
    """
    :param G: networkx
    :param spreadability_list:[(Node,Spreadability),(Node,Spreadability),...]
    :param k: seed size
    :return: seed nodes:[Node,Node,Node,...]
    """

    selected_nodes = []


    for node, _ in spreadability_list:
        if len(selected_nodes) >= k:
            break


        connections = sum(1 for neighbor in G.neighbors(node) if neighbor in selected_nodes)


        if connections == 0:
            selected_nodes.append(node)


    for node, _ in spreadability_list:
        if len(selected_nodes) >= k:
            break
        if node not in selected_nodes:
            selected_nodes.append(node)

    return selected_nodes