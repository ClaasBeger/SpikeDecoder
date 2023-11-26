import torch
from torch import functional as F
from torch import Tensor
import seaborn as sns
from spikingjelly.clock_driven.surrogate import ATan as atan
import matplotlib.pyplot as plt
import model


def position_encoding(
        seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    '''
    Returns the positional encoding tensor for a given token

    Parameters
    ----------
    seq_len : int
        Lenght of the input sequence (e.g "I am a Robot" -> 4).
    dim_model : int
        The dimension of the model, e.g the length of one positional embedding vector
        (for example, 4 if the embedding consists of 4 values has to be equal to 
         the meaning embedding).
    device : torch.device, optional
        Device to be set for computation. The default is torch.device("cpu").

    Returns
    -------
    Tensor
        The positional encoding vector for the given token. There is also the n 
        parameter, which is a user defined scalar. It has been set to 10000 here
        as was suggested in "Attention is all you need". Also, ordinarily a var
        i would be introduced which separates odd from even numbers, and always 
        couples one sine and one cosine step together. 

    '''
    
    # reshape to size 1*seq_len*1 ([[[1],[2],[3],[4],[5]]])
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    # reshape to size 1*1*seq_len ([[[0,1,2,3,4,5]]])
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    # calculate positions tensor / 10000^(dim vector / d_model) (scaling)
    phase = pos / (1e4 ** (dim / dim_model))
    
    # replace all pos encodings that are even with torch.sin(phase) and odd with torch.cos(phase)
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase)) 

def embedding_comparison():
    # figsize=(6, 6) control width and height
    # dpi = 600, I 
    plt.figure(figsize=(10, 6), dpi = 600) 
    colormap = sns.color_palette("RdYlGn_r", as_cmap=True)
    yticks = ['One-Hot', 'Binary', 'Learned', 'Learned*']
    xticks = ['One-Hot', 'Binary', 'Static', 'Learned']
    sizes = [[17780000, 330000, 910000, 920000], 
             [14820000, 40000, 360000, 140000],
             [20350000, 750000, 1420000, 1420000],
             [10195000, 10195000, 910000, 930000]]
    sizeAnnots = [[17780000, 330000, 910000, 920000], 
             [14820000, 40000, 360000, 140000],
             [20350000, 750000, 1420000, 1420000],
             [10195000, 10195000, 910000, 930000]]
    accuracies = [[96.3, 23.7, 41.2, 41.1], 
             [94.2, 23.7, 27.6, 21.5],
             [96.8, 33.3, 54.5, 52.5],
             ["X", "X", 44.3, 41.2]]
    accuracyValues = [[96.3, 23.7, 41.2, 41.1], 
             [94.2, 23.7, 27.6, 21.5],
             [96.8, 33.3, 54.5, 52.5],
             [0, 0, 44.3, 41.2]]
    noAccuracies = [["", "", "", ""], 
             ["", "", "", ""],
             ["", "", "", ""],
             ["X", "X", "", ""]]
    for row in range(len(accuracies)):
        for entry in range(len(accuracies[row])):
            if(not accuracies[row][entry] == "X"):
                accuracies[row][entry] = str(accuracies[row][entry])+"%"
        
    for row in range(len(sizes)):
        for entry in range(len(accuracies[row])):
            if(row==3 and (entry==0 or entry==1)):
                sizeAnnots[row][entry] = str('X')
            else:
                sizeAnnots[row][entry] = str(sizeAnnots[row][entry]/1000000)+'e6'
                
    # annot=accuracies
    axs = sns.heatmap(sizes, annot=sizeAnnots, annot_kws={"size": 12}, cmap = colormap,
                      fmt = '',xticklabels=xticks, yticklabels=yticks, linewidth=1)
    
    plt.tick_params(axis = 'x', labelsize = 12) # x font label size
    plt.tick_params(axis = 'y', labelsize = 12) # y font label size
    
    axs.set_xlabel('Position Embedding', fontsize=12)
    axs.set_ylabel('Token Embedding', fontsize=12)

def embedding_visualization(embedding, labels = []):
    a = atan()
    
    plt.figure(figsize=(10, 6), dpi = 600) 
    
    if(len(embedding.shape)>2):
        embedding = embedding[-1]
    predictions = embedding.tolist()
    
    colormap = sns.color_palette("YlOrBr", as_cmap=True)
    
    yticks = []
    xticks = []
    
    for idx in range(len(predictions)):
        yticks.append('Entry '+str(idx+1))
    
    yticks = labels
    
    if(len(yticks) > 0):
        axs = sns.heatmap(predictions, cmap = colormap, fmt = '',xticklabels=False, yticklabels=yticks, linewidth=1)
    else:
        axs = sns.heatmap(predictions, cmap = colormap, fmt = '',xticklabels=False, yticklabels=False, linewidth=1)
        
def display_Transition_graph(accuracies=None):
    fig, ax = plt.subplots()
    accuracies = [98.5, 98.4, 72.6, 92.2, 75.0, 73.0]
    accuracies_5head = [95.7, 82.0, 80.0]
    ax.plot([0,1,2,2,3,4], accuracies, color='red', label="8-10 Heads")
    ax.plot([2,3,4], accuracies_5head, color='green', label="5 Heads")
    ax.legend(loc = 'lower left')
    plt.title("Spike Transition")
    plt.xticks([0,1,2,3,4])
    plt.xlabel('Spike Degree')
    plt.ylabel('Accuracy')
    
    
def display_Transition_graph_v2(accuracies=None):
    fig, ax = plt.subplots(figsize=(10,6))
    accuracies = [98.5, 98.4, 72.6]
    accuracy_depth = [72.6,92.2,74.6,]
    accuracy_head_reduction = [74.6, 82.0, 78.5]
    accuracy_one_head = [78.5, 87.0]
    ax.plot([0,1,2], accuracies, color='red', label="6 Blocks, 8-10 Heads")
    ax.plot([2,2,3], accuracy_depth, color='green', label="12 Blocks, 8-10 Heads")
    ax.plot([3,3,4], accuracy_head_reduction, color='blue', label="12 Blocks 5 Heads")
    ax.plot([4,4], accuracy_one_head, color='black', label="12 Blocks 1 Head")
    ax.legend(loc = 'lower left')
    plt.title("Spike Transition")
    plt.xticks([0,1,2,3,4])
    plt.xlabel('Spike Degree')
    plt.ylabel('Accuracy')
