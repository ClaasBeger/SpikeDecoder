import model 
import NonSpikingDecoderModel as NSD
import math
import skorch
import torch
import torch.nn as nn
from torch import optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
import utils
import numpy as np


def accuracy_categorical(estimator, y, y_pred, **kwargs):
    '''
    

    Parameters
    ----------
    y : torch.Tensor
        The true labels of the predictions.
    y_pred : torch.Tensor
        The predictions made by the model.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    float
        The accuracy score.

    '''
    #y = y[:,-1,:] #Comment out for spiking variant
    
    y = estimator.module_.embedder(y, target=True, fullSequence=False)

    y_pred = y_pred.detach()
    
    
    return accuracy_score(y.flatten(), y_pred.flatten(), **kwargs)



adjusted_accuracy = make_scorer(accuracy_categorical)

reader = utils.Reader()


#data, targets, full = reader.generateRandomTextExcerpts(fullLength=5*26)

train_dataset = utils.charDataset('random', 'train', inputLength=256, fullLength=10000, seed=7070)
test_dataset = utils.charDataset('random', 'test', inputLength=256,  fullLength=10000, seed=7070)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Intializing skorch wrapper
model = skorch.NeuralNetClassifier(
    module=model.SpikingDecoderModel,#NSD.NonSpikingDecoderModel,
    train_split=None,
    batch_size = 8,
    lr = 0.0005,
    criterion= nn.CrossEntropyLoss,
    optimizer= optim.Adam,
    optimizer__weight_decay=0.1,
    optimizer__betas=(0.9,0.95),
    module__vocab_size = 30,
    module__encodingType = 'learned',
    module__position_encoding_strategy = 'static',
    module__max_len = 256,
    module__dim_hid = 16,
    module__mlp_dropout = 0.0,
    module__attn_dropout = 0.0,
    module__heads = 8,
    module__blocks = 6,
    module__tok_embed_dim = 50,
    module__timesteps = 4
)

param_grid = {
    'batch_size' : [32],
    'lr' : [0.0001],
    'module__mlp_dropout' : [0.0],
    'module__attn_dropout' : [0.0],
    'module__heads' : [1, 4, 8],
    'module__dim_hid' : [16],
    'module__blocks' : [1,4,8],
    'max_epochs' : [5],
    'module__tok_embed_dim' : [50]
    }

# slice(0, math.ceil(len(data)*0.9)), slice(math.ceil(len(data)*0.9), len(data)))
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=None, cv=[(slice(None), slice(None))], error_score = 'raise', scoring=adjusted_accuracy)
grid_result = grid.fit(train_dataset.fdata, train_dataset.ftargets)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))