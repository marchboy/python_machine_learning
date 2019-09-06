# -*- coding: utf-8 -*-

import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Image


G_karate = nx.karate_club_graph()
pos = nx.spring_layout(G_karate)
nx.draw(G_karate, cmap=plt.get_cmap('rainbow'), with_labels=True, pos=pos)


# 创建图形（没有节点和边）
G = nx.Graph()
H = nx.path_graph(10)


# 添加节点
G.add_node(1)
G.add_nodes_from([2,3])

# 添加一个边
G.add_edge(1,2)
e = (2,3)
G.add_edge(*e)

## 添加边列表
G.add_edges_from([(1, 2), (1, 3)])
G.add_edges_from(H.edges)

# 查看G的边和节点
G.number_of_edges()
G.number_of_nodes()



plt.show()


