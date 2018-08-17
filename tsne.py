import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import numpy as np
from utils import *
from sklearn.manifold import TSNE

def tsne(X, n_components):
    model = TSNE(n_components=2, perplexity=40)
    return model.fit_transform(X)

def plot_scatter(x, labels, title, txt = False):
    # colors = ['red', 'green', 'blue', '#F2F525', 'cyan']
    # data_class_name = ['Anger', 'Emphatic', 'Neutral', 'Positive', 'Rest']
    colors = ['red', 'green', '#F2F525', 'cyan']
    data_class_name = ['Anger', 'Emphatic', 'Positive', 'Rest']
    legend_patches = []
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    for i in range(len(colors)):
    	legend_patches.append(mpatches.Patch(color=colors[i], label=data_class_name[i]))
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    ax.scatter(x[:,0], x[:,1], c = labels , cmap = matplotlib.colors.ListedColormap(colors))    
    #plt.show()
    plt.savefig('./t-SNE/cnn_test_tsne_wo_N.png')



cnn_final = np.load('./t-SNE/cnn_test_hidden_output.npy')
label = np.load('./data/fau_test_label.npy')
A_index = []
for i in range(cnn_final.shape[0]):
    if np.array_equal(label[i], [1,0,0,0,0]) == True:
        A_index.append(i)
E_index = []
for i in range(cnn_final.shape[0]):
    if np.array_equal(label[i], [0,1,0,0,0]) == True:
        E_index.append(i)
N_index = []
for i in range(cnn_final.shape[0]):
    if np.array_equal(label[i], [0,0,1,0,0]) == True:
        N_index.append(i)
P_index = []
for i in range(cnn_final.shape[0]):
    if np.array_equal(label[i], [0,0,0,1,0]) == True:
        P_index.append(i)
R_index = []
for i in range(cnn_final.shape[0]):
    if np.array_equal(label[i], [0,0,0,0,1]) == True:
        R_index.append(i)

cnn_final = np.delete(cnn_final, A_index[200:], axis=0)
cnn_final = np.delete(cnn_final, E_index[200:], axis=0)
cnn_final = np.delete(cnn_final, N_index[200:], axis=0)
cnn_final = np.delete(cnn_final, P_index[200:], axis=0)
cnn_final = np.delete(cnn_final, R_index[200:], axis=0)
print (cnn_final.shape)
# cnn_final_wo_N = np.delete(cnn_final, N_index, axis=0)
# label_wo_N = np.delete(label, N_index, axis=0)
# label_wo_N = np.delete(label_wo_N, 2,axis=1)

# print (cnn_final_wo_N.shape)
# print (label_wo_N.shape)

# tsne = tsne(cnn_final_wo_N, 2)
# print (tsne.shape)
# label_wo_N = label_wo_N.astype(float)
# label2 = []
# for i in range (label_wo_N.shape[0]):
# 	label2.append(np.argmax(label_wo_N[i]))
# label2 = np.asarray(label2)
# plot_scatter(tsne, label2, '')

