import argparse
import torch
from torch.autograd import Variable
from torch import nn
from seq2seq.util.checkpoint import Checkpoint
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import matplotlib.mlab as mlab
import scipy
from mpl_toolkits.axes_grid1 import AxesGrid
from graph_tool.all import *
from pylab import *  # for plotting
from numpy.random import *  # for random sampling
from matplotlib.colors import Normalize
import random
from models import Model, Models, Analysis
import matplotlib.cm as cm

#array([[  0.,   3.,   6.],
#   [  9.,  12.,  15.],
#   [ 18.,  21.,  24.]])

cmap = cm.seismic
norm = Normalize(vmin=-1, vmax=1)

guided_gru = Models('../machine-zoo/guided/gru', 'Guided_GRU')
baseline_gru = Models('../machine-zoo/baseline/gru', 'Baseline_GRU')

# guided_lstm = Models('../machine-zoo/guided/lstm', 'Guided_LSTM')
# baseline_lstm = Models('../machine-zoo/baseline/lstm', 'Baseline_LSTM')


def viz(models,model_num,param_name, threshold=None):
    weight = models.models[model_num].params[param_name]
    weight = pd.DataFrame(weight)
    print('Shape of weight matrices', weight.shape)
    g = Graph()
    edge_color = g.new_edge_property('vector<double>')
    g.edge_properties['edge_color']=edge_color
    num_next = weight.shape[1]
    num_prev = weight.shape[0]
    edge_weights = g.new_edge_property('double')
    pos = g.new_vertex_property("vector<double>")
    ver = [g.add_vertex() for i in range(num_next)]
    for i in range(num_prev):
        v = g.add_vertex()
        pos[v] = (i*num_next, 0)
        for j in range(num_next):
            pos[ver[j]] = (j*num_prev, 0.7*(num_next*num_prev))
            curr_weight = weight.iloc[i,j]
            if threshold:
                if curr_weight < threshold[0] or curr_weight > threshold[1]:
                    e = g.add_edge(v, ver[j])
                    edge_color[e] = cmap(norm(curr_weight))
                    edge_weights[e] = curr_weight
            else:
                e = g.add_edge(v, ver[j])
                edge_color[e] = cmap(norm(curr_weight))
                edge_weights[e] = curr_weight

    g.edge_properties["weight"] = edge_weights
    image_folder = 'images/viz/'

    if threshold:
        filename = image_folder + models.title + '_' + model_num + '_' + param_name + '_' + str(threshold[0]) + '_'+ str(threshold[1]) +'.png'
    else:
        filename = image_folder + models.title  + '_' + model_num + '_' + param_num +  '.png'
    graph_draw(g,pos=pos, output=filename,output_size=[4024,4024],vertex_size=4,edge_color=g.edge_properties['edge_color'],edge_pen_width=prop_to_size(edge_weights, mi=-1, ma=1, power=1) )



#Baseline_GRU_3_encoder.rnn.weight_w_hz_-0.2_0.2.png
end = ['z', 'r']
param = 'encoder.rnn.weight_w_h'
threshold = [-0.3, 0.3]
model_num = 3
print(baseline_gru.models[str(1)].params.keys())
for e in end:
    viz(baseline_gru, str(model_num), param+e, threshold)
    viz(guided_gru, str(model_num), param+e, threshold)
