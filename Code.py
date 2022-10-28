import networkx as nx
import random as rnd
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import losses
from sklearn.model_selection import train_test_split

import time


def random_node(G):
    n = sorted(G.nodes())
    return n[rnd.randint(0, nx.number_of_nodes(G) - 1)]


def random_edge(G):
    e = sorted(G.edges())
    return e[rnd.randint(0, nx.number_of_edges(G) - 1)]


Dataset = "netscience.mtx"
split = " "
alpha = 0.6


def sigmoid(val):
    return 1 / (1 + math.exp(-val))


def compute(G, X_Edges):
    FeatureList = []
    for e in X_Edges:
        ## AA
        try:
            preds = nx.adamic_adar_index(G, [e])
            for u, v, p in preds:
                AA = p
        except:
            AA = 0

        ## AP
        try:
            preds = nx.preferential_attachment(G, [e])
            for u, v, p in preds:
                AP = p
        except:
            AP = 0

        ## CN
        CN = len(sorted(nx.common_neighbors(G, e[0], e[1])))

        ## HD
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = max(len(list(G.neighbors(e[1]))), len(list((G.neighbors(e[0])))))
        if r != 0:
            HD = x / r
        else:
            HD = 0
        ## HP
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = min(len(list(G.neighbors(e[1]))), len(list(G.neighbors(e[0]))))
        if r != 0:
            HP = x / r
        else:
            HP = 0

        ## JA
        try:
            preds = nx.jaccard_coefficient(G, [e])
            for u, v, p in preds:
                JA = p
        except:
            JA = 0

        ## LHN
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = len(list(G.neighbors(e[1]))) * len(list(G.neighbors(e[0])))
        if r != 0:
            LHN = x / r
        else:
            LHN = 0

        ## PD
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = math.pow(len(list(G.neighbors(e[1]))) * len(list(G.neighbors(e[0]))), 0.02)
        if r != 0:
            PD = x / r
        else:
            PD = 0
        ## RA
        try:
            preds = nx.resource_allocation_index(G, [e])
            for u, v, p in preds:
                RA = p
        except:
            RA = 0

        ##SA
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = math.sqrt(len(list(G.neighbors(e[1]))) * len(list(G.neighbors(e[0]))))
        if r != 0:
            SA = x / r
        else:
            SA = 0

        ## SO
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = (len(list(G.neighbors(e[1]))) + len(list(G.neighbors(e[0]))))
        if r != 0:
            SO = x / r
        else:
            SO = 0

        ## proposed metric
        d = nx.betweenness_centrality(G)

        preds = nx.resource_allocation_index(G, [e])
        for u, v, p in preds:
            PSI = alpha * math.pow(p, 1 - alpha) + sigmoid(d[e[0]] + d[e[1]])
        FeatureList.append([AA, AP, CN, HD, HP, JA, LHN, PD, RA, SA, SO, PSI])
    return FeatureList


def NN(G):
    '''
    In this function:
        0- we construct a set that contains 2*|E| where |E| are the links of G and the other |E| are edges not in G.
        1- we construct the feature vector of each link. This vector contains [AA,CN,JA,....] and y=1 when the link exists
            and 0 when not.
        2- we pass the features  to NN to train the model  and  then we  evaluate it using accuracy ...

    '''
    ### this    is  step 0
    G = G.copy()
    m = nx.number_of_edges(G)
    Y_edges = list()
    negativeExamples = set()
    while len(negativeExamples) < m:
        a = random_node(G)
        b = random_node(G)
        if a == b or a in nx.neighbors(G, b):
            continue
        negativeExamples.add((a, b))
        Y_edges.append(0)
    X_Edges = list(negativeExamples)

    # positiveExamples = []
    for e in G.edges():
        X_Edges.append(e)
        Y_edges.append(1)
    print("Generation  of X_Edges  and Y_edges is done !")
    # we shuffle the lists
    temps = list(zip(X_Edges, Y_edges))
    rnd.shuffle(temps)
    X_Edges, Y_edges = zip(*temps)

    print("shuffle is done ! the length of X_edges is=", len(X_Edges))

    ### this  is step 1
    FeatureList = []
    # i=0
    # k=10
    for e in X_Edges:

        ## AA
        try:
            preds = nx.adamic_adar_index(G, [e])
            for u, v, p in preds:
                AA = p
        except:
            AA = 0

        ## AP
        try:
            preds = nx.preferential_attachment(G, [e])
            for u, v, p in preds:
                AP = p
        except:
            AP = 0

        ## CN
        CN = len(sorted(nx.common_neighbors(G, e[0], e[1])))

        ## HD
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = max(len(list(G.neighbors(e[1]))), len(list((G.neighbors(e[0])))))
        if r != 0:
            HD = x / r
        else:
            HD = 0
        ## HP
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = min(len(list(G.neighbors(e[1]))), len(list(G.neighbors(e[0]))))
        if r != 0:
            HP = x / r
        else:
            HP = 0

        ## JA
        try:
            preds = nx.jaccard_coefficient(G, [e])
            for u, v, p in preds:
                JA = p
        except:
            JA = 0

        ## LHN
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = len(list(G.neighbors(e[1]))) * len(list(G.neighbors(e[0])))
        if r != 0:
            LHN = x / r
        else:
            LHN = 0

        ## PD
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = math.pow(len(list(G.neighbors(e[1]))) * len(list(G.neighbors(e[0]))), 0.02)
        if r != 0:
            PD = x / r
        else:
            PD = 0
        ## RA
        try:
            preds = nx.resource_allocation_index(G, [e])
            for u, v, p in preds:
                RA = p
        except:
            RA = 0

        ##SA
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = math.sqrt(len(list(G.neighbors(e[1]))) * len(list(G.neighbors(e[0]))))
        if r != 0:
            SA = x / r
        else:
            SA = 0

        ## SO
        x = len(sorted(nx.common_neighbors(G, e[0], e[1])))
        r = (len(list(G.neighbors(e[1]))) + len(list(G.neighbors(e[0]))))
        if r != 0:
            SO = x / r
        else:
            SO = 0

        ## proposed metric
        '''
        this metric sould be commented in order to compare the performance of NN with and without PSI
        '''
        d = nx.betweenness_centrality(G)

        preds = nx.resource_allocation_index(G, [e])
        for u, v, p in preds:
             PSI = alpha * math.pow(p, 1 - alpha) + sigmoid(d[e[0]] + d[e[1]])
        FeatureList.append([AA, AP, CN, HD, HP, JA, LHN, PD, RA, SA, SO,PSI])
    print("Feature list generation is done!")
    # x_train, x_test, y_train, y_test = train_test_split(FeatureList, Y_edges, test_size=0.1)
    X_Edges = np.array(FeatureList)
    # print(X_Edges)
    Y_edges = np.array(Y_edges)
    # print(Y_edges)
    ### neural net design
    x_train = tf.keras.utils.normalize(X_Edges, axis=1)

    layer_sizes=[16,32,64,128]
    for layer_size1 in layer_sizes:
        for layer_size2 in layer_sizes:
                NAME="PSI-{}-L1-{}-L2-{}-{}".format(Dataset,layer_size1,layer_size2,int(time.time()))
                print(NAME)
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(layer_size1, activation=tf.nn.relu))
                model.add(tf.keras.layers.Dense(layer_size2, activation=tf.nn.relu))
                model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs26102022/{}".format(NAME))
                    #TensorBoard(log_dir="logs/{}_PSI_{}".format(Dataset, int(time.time())))
                ##binary_crossentropy  sparse_categorical_crossentropy
                model.compile(
                     optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
                 # print(x_train)
                model.fit(X_Edges, Y_edges, epochs=50, callbacks=[tensorboard], validation_split=0.2)
                #model.fit(x_train, Y_edges, epochs=50, callbacks=[tensorboard], validation_split=0.2)
                val_loss, val_acc = model.evaluate(X_Edges, Y_edges)
                print(val_loss, "   ", val_acc)


rnd.seed(0)
G = nx.Graph()
f0 = open(Dataset)
text = f0.readlines()
i = 0
while i < text.__len__():
    line = text[i].split(split)
    u, v = int(line[0]), int(line[1])
    e = (u, v)
    G.add_edge(*e)
    i += 1
f0.close()
NN(G)
