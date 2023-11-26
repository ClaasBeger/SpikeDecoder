# -*- coding: utf-8 -*-
"""
Created on Thu May 11 19:33:58 2023

@author: claas
"""

import numpy as np
import nltk
from tqdm import tqdm
import torch
from torch import Tensor


class TensorEncoder():
    def __init__(self, vocab_path, data, embedding_dim:int, data_type="trian", bias=3) -> None:
        super(TensorEncoder, self).__init__()
        self.vocab_path = vocab_path
        self.data = data
        self.embedding_dim = embedding_dim
        self.data_type = data_type
        self.bias = bias

    def encode(self):
        """
        Loads and normalizes the pretrained weights from GloVe, through applying 
        the technique described in Spiking Convolutional Neural Networks for Text
        Classification.

        Returns
        -------
        embedding_tuple_list : List of embedding vectors for the respective tokens.

        """
        glove_dict = {}
        with open(self.vocab_path, "r", encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                glove_dict[word] = vector
        
        print(glove_dict["the"])

        mean_embedding = np.mean(np.array(list(glove_dict.values())), axis=0)
        zero_embedding = np.array([0] * self.embedding_dim, dtype=float)
        mean_value = np.mean(list(glove_dict.values()))
        variance_value = np.var(list(glove_dict.values()))
        left_boundary = mean_value - self.bias * np.sqrt(variance_value)
        right_boundary = mean_value + self.bias * np.sqrt(variance_value)
        
        self.data = self.data.split()
        
        print(self.data)

        embedding_tuple_list = []

        for j in range(len(self.data)):
                    word = self.data[j]
                    embedding = glove_dict[word] if word in glove_dict.keys() else zero_embedding
                    # N(0, 1)
                    embedding_n01 = (embedding - np.array([mean_value] * self.embedding_dim)) / np.array([np.sqrt(variance_value)] * self.embedding_dim)
                    embedding_norm = np.array([0] * self.embedding_dim, dtype=float)
                    for k in range(self.embedding_dim):
                        if embedding[k] < left_boundary:
                            embedding_norm[k] = -self.bias
                        elif embedding[k] > right_boundary:
                            embedding_norm[k] = self.bias
                        else:
                            embedding_norm[k] = embedding_n01[k]
                    # add abs(left_embedding)
                    embedding_norm = (embedding_norm + np.array([np.abs(self.bias)] * self.embedding_dim))/(self.bias * 2)
                    # embedding_norm = np.clip(embedding_norm, a_min=0, a_max=1)
                    embedding_tuple_list.append(embedding_norm)
        # print(i, sent_embedding)
                
        return embedding_tuple_list
    
    def position_encoding(
            max_len: int, dim_model: int, device: torch.device = torch.device("cpu")) -> Tensor:
        '''
        Returns the positional encoding tensor for a given token

        Parameters
        ----------
        seq_len : int
            Maximum length of the input sequence (e.g "I am a Robot" -> 4).
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
        pos = torch.arange(max_len, dtype=torch.float, device=device).reshape(1, -1, 1)
        # reshape to size 1*1*seq_len ([[[0,1,2,3,4,5]]])
        dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
        # calculate positions tensor / 10000^(dim vector / d_model) (scaling)
        phase = pos / (1e4 ** (dim / dim_model))
        
        # replace all pos encodings that are even with torch.sin(phase) and odd with torch.cos(phase)
        return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
    
tensorEncoder = TensorEncoder(vocab_path="../Data/glove.6B.50d.txt", data="This is a test", embedding_dim=50)

print(tensorEncoder.encode())
    
    
