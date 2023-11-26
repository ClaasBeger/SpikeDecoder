# -*- coding: utf-8 -*-
"""
Created on Thu May  4 04:24:14 2023

@author: Claas Beger
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from spikingjelly.clock_driven.surrogate import ATan as atan
from spikingjelly.clock_driven import functional
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import argmax
import math
import re
from PowerNorm import MaskPowerNorm
if(torch.cuda.is_available()):
    import cupy

#TODO: add documentation for dictionary and frequency
    
class SDconfig:
    """
    The spiking Decoder config class.
    Most of the Model specific hyperparameters are listed here
    """
    def __init__(
            self, vocab_size, max_len, encodingType: str = 'two-hot',
            position_encoding_strategy : str = "linear",
            tok_embed_dim=30, pos_embed_dim = 30, attn_dropout = 0.0, mlp_dropout = 0.0,
            heads=2, blocks=1, spiking_residuals: int = 0, spiking_head : bool = True,
            timesteps : int = 2, device = torch.device('cpu'), attention_scale : float = 0.125,
            normalization = 'layer', combine_position : bool = False,
            firstBlock_spiking_res : bool = False, spike_mode : str = 'average',
            pos_bias : float = .0, float_reunite : bool = False,
            float_embedding : bool = False, learning_MSLIF : bool = False,
             dictionary : list = None, track_frequency: bool = False,
            **kwargs
            ):
        """
        

        Parameters
        ----------
        vocab_size : int
            The size of the full vocabulary (including special tokens such as Start, End and Padding.)
        max_len : int
            Max length of the input sequence by character count (excl. Start and End Token).
        encodingType : str
            The representation strategy for the embedding, should be either 'one-hot','two-hot','binary', or 'learned'.
            The dafult is 'two-hot'.
        tok_embed_dim : int, optional
            The number of elements representing one element from the vocabulary.
            Excluding the positional encoding, only has an effect for variable length
            encodingType learned. The default is 30.
        pos_embed_dim : int, optional
            The number of elements representing the position of one element from the vocabulary.
            Only has an effect for variable length encodingTypes such as static or learned.
            The default is 30.
        attn_dropout : Float, optional
            The dropout to be applied on the attention matrix product. The default
            is 0.0.
        mlp_dropout : Float, optional
            The dropout to be applied inside the MLP. The default is 0.0.
        heads : int, optional
            The number of heads to be used in SSA. The default is 2.
        blocks : int, optional
            The number of chained decoder blocks to generate for the model.
        spiking_residuals : int, optional
            The degree to make the residuals in the decoder block consist of spike accumulation
            only. Degree 1 adjusts the residuals around the MLP. Degree 2 around the SMHA.
            The default is 0.
        spiking_head : bool, optional
            Whether to make the classification head spiking. Setting this to false improves
            performance, but introduces float MAC operations. The default is True.
        timesteps : int, optinal
            The number of synthetic timesteps to create during the embedding process.
            Increasing this number will improve performance, but slow down the model.
            The default is 2.
        device : torch.device, optional
            The device to create the tensors on. The default is torch.device('cpu')
        attention_scale : float, optional
            The value to scale the self-attention results with. The default is 0.125.
        normalization : str, optional
            The normalization layers to employ in the model. Possible choices are
            'batch', 'layer' and 'PowerNorm'. Layer is the fastest and produces the
            most stable results, but as opposed to the other two, it is not spike-compatible.
            The default is layer.
        combine_position : bool, optional
            Whether to sum up token and positional embedding rather than concatenate it.
            This is only possible for float-based embeddings of equal length,
            meaning learned and learned/static. If combined, the values will be mapped
            to binary value range after combination.
            The default value is False.
        firstBlock_spiking_res : bool, optional
            Whether to adjust the residual around the very first Block to circumvent
            spike and float addition. This issue only appears if binary embedding is used
            The default value is False.
        spike_mode : str, optional
            The strategy to recombine synthetic timesteps prior to prediction.
            Possible choices are 'average', 'accumulate' and 'concatenate'.
            The default value is 'average'.
        pos_bias : float, optional
            A bias to subtract from static positional embedding values prior to 
            binary mapping, in order to correct the tendency to include more ones than zeros.
            The default value is 0.
        float_reunite : bool, optional
            Whether to reunite timesteps on float values, prior to binary mapping in
            the classification head. This may incur spike-incompatible operations.
            The default value is False.
        float_embedding : bool, optional
            Whether to disable the mapping to binary value range of learned and static 
            embeddings, prior to computation inside the model. 
            The default value is False.
        learning_MSLIF : bool, optional
            Whether to use MultistepLIFs with learnable threshold. The default value is False.
        **kwargs : dict
            Potential additonal params.

        Returns
        -------
        The created SDconfig object.

        """
        
        assert position_encoding_strategy in ['linear', 'binary', 'learned', 'static']
        
        self.vocab_size = vocab_size
        self.max_len = max_len 
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.num_blocks = blocks
        self.num_heads = heads
        self.encodingType = encodingType
        self.position_encoding_strategy = position_encoding_strategy
        self.spiking_residuals = spiking_residuals
        self.spiking_head = spiking_head
        self.timesteps = timesteps
        self.dictionary = dictionary
        if(dictionary != None):
            self.vocab_size = len(dictionary)
            vocab_size = len(dictionary)
        self.combine_position = combine_position
        self.tok_embed_dim = tok_embed_dim
        self.float_embedding = float_embedding
        if(self.combine_position):
            self.pos_embed_dim = tok_embed_dim
        else:
            self.pos_embed_dim = pos_embed_dim
        self.attention_scale = attention_scale
        self.normalization = normalization
        self.firstBlock_spiking_res = firstBlock_spiking_res
        self.spike_mode = spike_mode
        self.pos_bias = pos_bias
        self.float_reunite = float_reunite
        self.learning_MSLIF = learning_MSLIF
        self.track_frequency = track_frequency
        if(self.track_frequency):
            self.smha_in_nnz = []
            self.query_nnz = []
            self.att_nnz = []
            self.smha_out_nnz = []
            self.mlp_in_nnz = []
            self.mlp_out_nnz = []
            self.head_nnz = []
        if(encodingType=='one-hot'):
            self.embed_dim = vocab_size*self.max_len
        elif(encodingType=='two-hot'):
            if(position_encoding_strategy == 'binary'):
                self.embed_dim = vocab_size+math.ceil(math.log2(max_len))
            elif(position_encoding_strategy in ['static', 'learned']):
                self.embed_dim = self.vocab_size+self.pos_embed_dim
            else:
                self.embed_dim = vocab_size+self.max_len
        elif(encodingType=='binary'):
            if(position_encoding_strategy == 'binary'):
                self.embed_dim = math.ceil(math.log2(vocab_size))+math.ceil(math.log2(max_len))
            elif(position_encoding_strategy in ['static', 'learned']):
                self.embed_dim = math.ceil(math.log2(vocab_size))+self.pos_embed_dim
            else:
                self.embed_dim = math.ceil(math.log2(vocab_size))+self.max_len
        else:
            if(position_encoding_strategy == 'linear'):
                self.embed_dim = tok_embed_dim+self.max_len
            elif(position_encoding_strategy in ['static', 'learned']):
                if(self.combine_position):
                    self.embed_dim = tok_embed_dim
                else:
                    self.embed_dim = tok_embed_dim+self.pos_embed_dim
            elif(position_encoding_strategy == 'binary'):
                self.embed_dim = tok_embed_dim+math.ceil(math.log2(max_len))
        self.device = device
        if(device==torch.device('cpu') or self.learning_MSLIF):
            # registering learnable parameters does not work with cupy
            self.backend = 'torch'
        else:
            self.backend = 'cupy'
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __str__(self):
            return """SD config with vocab size {vocab_size},
        max length {max_length}, 
        number of blocks {blocks}, 
        number of heads {heads},
        position encoding strategy {pos_strategy},
        token embedding of type {tok_embedding},
        spiking residuals set to {spiking_residuals},
        token embedding dimension set to {tok_embed_dim},
        position embedding dimension (if applicable) set to {pos_embed_dim},
        number of timesteps {timesteps},
        self attention scaling with {attention_scaling},
        normalization of type {normalization},
        combine position set to {combine_position},
        first Block spiking residual is set to {firstBlock_spiking_res},
        spike mode set to {spike_mode},
        static position encoding bias set to {pos_bias},
        float reunion set to {float_reunite},
        float embedding set to {float_embedding},
        learning MSLIF set to {learning_MSLIF},
        and spiking head set to {spiking_head}.""".format(vocab_size=self.vocab_size,
        max_length=self.max_len, blocks=self.num_blocks, heads=self.num_heads,
        pos_strategy=self.position_encoding_strategy, tok_embedding=self.encodingType,
        spiking_residuals=self.spiking_residuals, tok_embed_dim=self.tok_embed_dim, 
        timesteps=self.timesteps, attention_scaling=self.attention_scale, 
        normalization=self.normalization, combine_position=self.combine_position,
        firstBlock_spiking_res=self.firstBlock_spiking_res, 
        spiking_head=self.spiking_head, pos_bias=self.pos_bias,
        spike_mode=self.spike_mode, float_reunite=self.float_reunite,
        float_embedding=self.float_embedding, learning_MSLIF=self.learning_MSLIF,
        pos_embed_dim=self.pos_embed_dim)
            

class SpikingInputEmbeddingBlock(nn.Module):
    """
    The Embedding Block to process and encode the input characters.
    """
    
    # Remove this annotation in case you want learned embedding
    #@torch.no_grad()
    def __init__(self, config, create_timestep: bool = True):
        """
        

        Parameters
        ----------
        config : SDconfig
            The config object which denotes the model hyperparameters.
        reducedAlphabet : bool
            Whether to reduce the alphabet size by symbols.
        position_encoding_strategy : str
            The encoding strategy to apply on positional information.
        create_timestep: bool, optional
            Whether to create the timestep dimension, which may be avoided in 
            the partially spiking decoder model. The default is True.

        Returns
        -------
        The created SpikingInputEmbeddingBlock object.

        """
        super().__init__()
        
        self.position_encoding_strategy = config.position_encoding_strategy
       
        self.config = config
        
        self.create_timestep = create_timestep
        
        if(self.position_encoding_strategy == 'binary'):
            self.pos_length = math.ceil(math.log2(config.max_len))
        elif(self.position_encoding_strategy == 'static'):
            self.pos_table = self.position_encoding(self.config.max_len, self.config.pos_embed_dim)
            self.binmapping = atan()
        elif(self.position_encoding_strategy == 'learned'):
            self.pos_table = torch.nn.Embedding(self.config.max_len, self.config.pos_embed_dim, device=self.config.device
            )
            self.pos_table.weight.data.normal_(mean=0.0, std=1e-5)
            if(self.config.float_embedding):
                self.binmapping = lambda x: x 
            else:
                self.binmapping = atan()
        
        self.config = config
        

        # define universe of possible input values
        # Adjust for full word embedding option
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz.\'SEP '
        if(self.config.vocab_size==30):
            self.alphabet = 'abcdefghijklmnopqrstuvwxyz.\'P '
            
        if(self.config.dictionary != None):
            self.alphabet = self.config.dictionary
        
        if(config.encodingType=='binary'):
            self.enc_length = math.ceil(math.log2(len(self.alphabet)))

        # define a mapping of chars to integers
        self.tok_to_int = dict((c, i) for i, c in enumerate(self.alphabet))
        self.int_to_tok = dict((i, c) for i, c in enumerate(self.alphabet))
        
        if(config.encodingType=='learned'):
            self.tok_embed = nn.Embedding(
            self.config.vocab_size, self.config.tok_embed_dim, device=self.config.device
            )
            self.tok_embed.weight.data.normal_(mean=0.0, std=1e-5)
            if(self.config.float_embedding and self.binmapping==None):
                self.binmapping = lambda x: x 
            else:
                self.binmapping = atan()
        
        if(not (config.encodingType=='learned' or self.position_encoding_strategy=='learned')):
            self.requires_grad_(False)
        

            
    def encode(self, InputSequence, target:bool=False, fullSequence:bool=False):
        """
        

        Parameters
        ----------
        InputSequence : str or list
            The sequence to be encoded.
        target : bool, optional
            Denotes whether this is part of encoding a target sequence.
            If so, positional encoding will be skipped. The default is False.
        fullSequence : bool
            Whether the encoded input is a full Sequence (e.g should include Start 
            and End-token), the default is True.

        Returns
        -------
        result : torch.Tensor
            The encoded input vector.

        """
        result = list()
        if(fullSequence):
            InputSequence = 'S' + InputSequence + 'E'
        if(target==True):
            integer_encoded = [self.tok_to_int[tok] for tok in InputSequence] 
            result = torch.as_tensor(integer_encoded, device=self.config.device)
            # Add Batch dimension
            result = result.unsqueeze(0)
            return result
        # If neccessary, add padding to max len with zero vectors
        while(len(InputSequence)<self.config.max_len):
            if(self.config.dictionary==None):
                InputSequence = InputSequence + 'P'
            else:
                if(type(InputSequence) == np.ndarray):
                  InputSequence = np.append(InputSequence,'P')
                else:
                  InputSequence = InputSequence + ['P']
        if(self.config.encodingType=='two-hot'):
           integer_encoded = [self.tok_to_int[tok] for tok in InputSequence] 
           twohot_encoded = list()
           for (i, value) in enumerate(integer_encoded):
              letter = [0. for _ in range(len(self.alphabet))]
              letter[value] = 1.
              if(self.position_encoding_strategy == 'linear'):
                  position = [0. for _ in range(self.config.max_len)]
                  position[i] = 1.
              elif(self.position_encoding_strategy == 'binary'):
                  position = np.array([float(x) for x in bin(i)[2:]])
                  position.resize(self.pos_length)
                  position = position.tolist()
              elif(self.position_encoding_strategy == 'static'):
                  position = [x for x in self.binmapping(self.pos_table[0][i]-self.config.pos_bias).tolist()]
              elif(self.position_encoding_strategy == 'learned'):
                  position = [x for x in self.binmapping(self.pos_table(torch.tensor([i], device=self.config.device).int())).tolist()[0]]
              letter = letter + position
              tensor = torch.as_tensor(letter, device=self.config.device)
              twohot_encoded.append(tensor)
           result = torch.stack(twohot_encoded)
        elif(self.config.encodingType=='one-hot'):
            integer_encoded = [self.tok_to_int[tok] for tok in InputSequence] 
            onehot_encoded = list()
            for (i, value) in enumerate(integer_encoded):
               letter = [0 for _ in range(self.config.vocab_size)]
               letter[value] = 1
               lst = [[0]*self.config.vocab_size for _ in range(self.config.max_len)]
               lst[i] = letter
               flattened_lst = [item for sublist in lst for item in sublist]
               tensor = torch.as_tensor(flattened_lst, device=self.config.device)
               onehot_encoded.append(tensor)
            result = torch.stack(onehot_encoded)
        elif(self.config.encodingType=='binary'):
            integer_encoded = [self.tok_to_int[tok] for tok in InputSequence] 
            binary_encoded = list()
            for (i, value) in enumerate(integer_encoded):
               letter = np.array([float(x) for x in bin(value)[2:]])
               letter.resize(self.enc_length)
               letter = letter.tolist()
               if(self.position_encoding_strategy == 'linear'):
                   position = [0. for _ in range(self.config.max_len)]
                   position[i] = 1.
               elif(self.position_encoding_strategy == 'binary'):
                   position = np.array([float(x) for x in bin(i)[2:]])
                   position.resize(self.pos_length)
                   position = position.tolist()
               elif(self.position_encoding_strategy == 'static'):
                   position = [x for x in self.binmapping(self.pos_table[0][i]-self.config.pos_bias).tolist()]
               elif(self.position_encoding_strategy == 'learned'):
                   position = [x for x in self.binmapping(self.pos_table(torch.tensor([i], device=self.config.device).int())).tolist()[0]]
               letter = letter + position
               tensor = torch.as_tensor(letter, device=self.config.device)
               binary_encoded.append(tensor)
            result = torch.stack(binary_encoded)
        elif(self.config.encodingType=='learned'):
            encodedInput = torch.as_tensor([int(self.tok_to_int[tok]) for tok in InputSequence], device=self.config.device).int()
            result = self.tok_embed(encodedInput)
            integer_encoded = [self.tok_to_int[tok] for tok in InputSequence]
            positions = []
            for (i, value) in enumerate(InputSequence):
                if(self.position_encoding_strategy == 'linear'):
                    position = [0. for _ in range(self.config.max_len)]
                    position[i] = 1.
                elif(self.position_encoding_strategy == 'binary'):
                    position = np.array([float(x) for x in bin(i)[2:]])
                    position.resize(self.pos_length)
                    position = position.tolist()
                elif(self.position_encoding_strategy == 'static'):
                    if(self.config.combine_position):
                        position = [x for x in self.pos_table[0][i].tolist()]
                    else:
                        position = [x for x in self.binmapping(self.pos_table[0][i]-self.config.pos_bias).tolist()]
                elif(self.position_encoding_strategy == 'learned'):
                    if(self.config.combine_position):
                        position = [x for x in self.pos_table(torch.tensor([i], device=self.config.device).int()).tolist()[0]]
                    else:
                        position = [x for x in self.binmapping(self.pos_table(torch.tensor([i], device=self.config.device).int())).tolist()[0]]
                positions.append(position)
            if(self.config.combine_position):
                result = result+torch.as_tensor(positions, device=self.config.device)
                result = self.binmapping(result)
            else:
                learned_encoded = self.binmapping(result)
                result = torch.cat((learned_encoded,torch.as_tensor(positions, device=self.config.device)), dim=1)
        if(self.create_timestep):
            # Stack to create timestep dimension
            result = torch.stack([result]*self.config.timesteps)
        # Add Batch dimension
        result = result.unsqueeze(0)
        return result
    
    def forward(self, InputSequence:list, target:bool=False, fullSequence:bool=False, **kwargs):
        '''
        

        Parameters
        ----------
        InputSequence : list
            The list of inputs, for example ['abc', 'hel'].
        target : bool, optional
            Denotes whether this is part of encoding a target sequence.
            If so, positional encoding will be skipped. The default is False.
        fullSequence : bool
            Whether the encoded input is a full Sequence (e.g should include Start 
            and End-token), the default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        torch.Tensor
            A tensor of the encoded inputs. Inputs will be concatenated under the 
            Batch dimension.

        '''
        encodedTokens = []
        for token in InputSequence: 
            encodedTokens.append(self.encode(token, target, fullSequence))
        encodedTokens = torch.cat(encodedTokens)
        if(self.create_timestep and not target):
            # transpose batch and timestep to conform to snn data input format
            encodedTokens = encodedTokens.transpose(0,1)
        return encodedTokens
    
    @torch.no_grad()
    def position_encoding(
            self, seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")) -> Tensor:
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
    
    @torch.no_grad()
    def showPrediction(self, pred:Tensor, target:str = None, batch_idx: int = None, is_proba: bool = False,
                       path : str = None, indexOffset : int = 0, top_k : int = 3):
        """
        
        Returns a Visualization of the prediction tensor in the form of a seaborn heatmap

        Parameters
        ----------
        pred : Tensor
            The tensor containing the prediction values.
        target : str, optional
            The string containing the target values. The default is None.
        batch_idx : int, optional
            Whether to visualize a specific batch, if more than one
            are given. The default is None.
        is_proba : bool, optional
            Whether the passed predictions are probability values. The default is False.
        path : str, optional
            Path to save the prediction visualization under. The default is None.
        indexOffset : int, optional
            Index offset by which to increase the first index value in the visualization.
            The default is 0.
        top_k : int, optional
            Number of tokens to display with highest model probability. Only applicable
            to Word Embeddings.
            The default is 3.
        Returns
        -------
        None.

        """
        
        # If contained, remove timestep dimension at position 2
        if(len(pred.shape) == 4):
            pred = pred[:,-1,:,:]
        if(len(pred.shape) ==  2):
            pred = pred.unsqueeze(0)
            
        if(not is_proba):
            pred = F.softmax(pred, -1)
        
        if(batch_idx != None):
            if(self.config.dictionary!=None):
                predictions = pred[batch_idx][-1]
            else:
                predictions = pred[batch_idx][-1].tolist()
        else:
            if(self.config.dictionary!=None):
                predictions = pred[-1]
            else:
                predictions = pred[-1].tolist()
        
        colormap = sns.color_palette("YlOrBr", as_cmap=True)
        
        yticks = []
        
        for idx in range(indexOffset, len(predictions)+indexOffset):
            yticks.append('index '+str(idx+1))
            
        if(self.config.dictionary != None):
            # pred dimensions [1, create_visuals_up_to/2, vocab]
            k_high_indices = torch.topk(predictions, top_k, -1)
            
            k_high_words = [[self.int_to_tok[index] for index in row] for row in k_high_indices[1].tolist()]
            
            axs = sns.heatmap(k_high_indices[0], annot=k_high_words, cmap = colormap, fmt = '',xticklabels=False, yticklabels=yticks, linewidth=1)
        else:
            axs = sns.heatmap(predictions, annot=[[char for char in self.alphabet]]*len(predictions), cmap = colormap, fmt = '',xticklabels=False, yticklabels=yticks, linewidth=1)
        
        if(target):
            for idx in range(len(target)):
                x_start = self.tok_to_int[target[idx]]
                axs.hlines(y = [idx], xmin=x_start, xmax = x_start+1, colors='green')
                axs.hlines(y = [idx+1], xmin=x_start, xmax = x_start+1, colors='green')
                axs.vlines(x = [x_start], ymin=idx, ymax = idx+1, colors='green')
                axs.vlines(x = [x_start+1], ymin=idx, ymax = idx+1, colors='green')
        
        if(path):
            plt.savefig(path)
            plt.clf()
        
# =============================================================================
#         top_k = torch.topk(pred, top, sorted=True).indices.squeeze()
#         d = {}
#         d[0] = [target, 100]
#         for top in range(len(top_k)):
#             d[top+1] = [self.tok_to_int[top_k[top].item()], pred[0][top_k[top].item()]]
#         print ("{:<8} {:<15} {:<10}".format('Pos','Char','Score'))
#         for k, v in d.items():
#             Char, Score = v
#             print ("{:<8} {:<15} {:<10}".format(k, Char, Score))
# =============================================================================

class SpikingDecoderModel(nn.Module):
    """
    The Spiking Decoder Model. 
    """
    
    def __init__(self, config=None, vocab_size=None, max_len=None, 
                 encodingType: str = 'two-hot', position_encoding_strategy: str = 'static',
                 dim_hid=None, attn_dropout=0.,
                 mlp_dropout = 0., heads=2, blocks=1, tok_embed_dim=30,
                 pos_embed_dim = 30, spiking_residuals: int = 0, 
                 spiking_head : bool = True, timesteps : int = 2, 
                 device = torch.device('cpu'), attention_scale : float = 0.125,
                 normalization = 'layer', combine_position : bool = False,
                 firstBlock_spiking_res : bool = False, spike_mode : str = 'average',
                 pos_bias : float = .0, float_reunite : bool = False,
                 float_embedding : bool = False, learning_MSLIF : bool = False
                 ):
        """
        

        Parameters
        ----------
        config: SDconfig
            The configuration object of the model. Optional, but if not passed, vocab_size
            and max_len have to be passed. 
         vocab_size : int
             The size of the full vocabulary (including special tokens such as Start, End and Padding.)
         max_len : int
             Max length of the input sequence by character count (excl. Start and End Token).
         encodingType : str
             The representation strategy for the embedding, should be either 'one-hot','two-hot','binary', or 'learned'.
             The dafult is 'two-hot'.
         tok_embed_dim : int, optional
             The number of elements representing one element from the vocabulary.
             Excluding the positional encoding, only has an effect for variable length
             encodingType learned. The default is 30.
         pos_embed_dim : int, optional
             The number of elements representing the position of one element from the vocabulary.
             Only has an effect for variable length encodingTypes such as static or learned.
             The default is 30.
         attn_dropout : Float, optional
             The dropout to be applied on the attention matrix product. The default
             is 0.0.
         mlp_dropout : Float, optional
             The dropout to be applied inside the MLP. The default is 0.0.
         heads : int, optional
             The number of heads to be used in SSA. The default is 2.
         blocks : int, optional
             The number of chained decoder blocks to generate for the model.
         spiking_residuals : int, optional
             The degree to make the residuals in the decoder block consist of spike accumulation
             only. Degree 1 adjusts the residuals around the MLP. Degree 2 around the SMHA.
             The default is 0.
         spiking_head : bool, optional
             Whether to make the classification head spiking. Setting this to false improves
             performance, but introduces float MAC operations. The default is True.
         timesteps : int, optinal
             The number of synthetic timesteps to create during the embedding process.
             Increasing this number will improve performance, but slow down the model.
             The default is 2.
         device : torch.device, optional
             The device to create the tensors on. The default is torch.device('cpu')
         attention_scale : float, optional
             The value to scale the self-attention results with. The default is 0.125.
         normalization : str, optional
             The normalization layers to employ in the model. Possible choices are
             'batch', 'layer' and 'PowerNorm'. Layer is the fastest and produces the
             most stable results, but as opposed to the other two, it is not spike-compatible.
             The default is layer.
         combine_position : bool, optional
             Whether to sum up token and positional embedding rather than concatenate it.
             This is only possible for float-based embeddings of equal length,
             meaning learned and learned/static. If combined, the values will be mapped
             to binary value range after combination.
             The default value is False.
         firstBlock_spiking_res : bool, optional
             Whether to adjust the residual around the very first Block to circumvent
             spike and float addition. This issue only appears if binary embedding is used
             The default value is False.
         spike_mode : str, optional
             The strategy to recombine synthetic timesteps prior to prediction.
             Possible choices are 'average', 'accumulate' and 'concatenate'.
             The default value is 'average'.
         pos_bias : float, optional
             A bias to subtract from static positional embedding values prior to 
             binary mapping, in order to correct the tendency to include more ones than zeros.
             The default value is 0.
         float_reunite : bool, optional
             Whether to reunite timesteps on float values, prior to binary mapping in
             the classification head. This may incur spike-incompatible operations.
             The default value is False.
         float_embedding : bool, optional
             Whether to disable the mapping to binary value range of learned and static 
             embeddings, prior to computation inside the model. 
             The default value is False.
        learning_MSLIF : bool, optional
             Whether to use MultiStepLIFNodes with learnable threshold. The default is False.
        dim_hid : int, optional
            The hidden dimension in the MLP. The default is None.
        dim_out : int, optional
            The out dimension in the MLP. The default is None.

        Returns
        -------
        The SpikingDecoderModel object.

        """
        super().__init__()
        
        assert (config != None or (vocab_size != None and max_len != None)), 'Either config or embedding dimension and max_len must be passed.'
        if(config != None):
           self.config = config 
        else:
            self.config = SDconfig(vocab_size, max_len, encodingType, position_encoding_strategy=position_encoding_strategy,
                                   pos_embed_dim=pos_embed_dim,
                                   attn_dropout=attn_dropout, mlp_dropout=mlp_dropout,
                                   heads=heads, blocks=blocks, tok_embed_dim=tok_embed_dim,
                                   spiking_residuals=spiking_residuals, 
                                   spiking_head=spiking_head, timesteps=timesteps, 
                                   device = device, attention_scale=attention_scale,
                                   normalization = normalization, combine_position=combine_position,
                                   firstBlock_spiking_res = firstBlock_spiking_res, 
                                   spike_mode = spike_mode, pos_bias = pos_bias,
                                   float_reunite = float_reunite, float_embedding = float_embedding,
                                   learning_MSLIF=learning_MSLIF)
        self.embedder = SpikingInputEmbeddingBlock(self.config)
        if(self.config.firstBlock_spiking_res):
            self.blocks = nn.Sequential(SpikingDecoderBlock(self.config.embed_dim, self.config, self.config.num_heads, dim_hid, firstBlock=True),*[SpikingDecoderBlock(self.config.embed_dim, self.config, self.config.num_heads, dim_hid) for b in range(self.config.num_blocks-1)])
        else:
            self.blocks = nn.Sequential(*[SpikingDecoderBlock(self.config.embed_dim, self.config, self.config.num_heads, dim_hid) for b in range(self.config.num_blocks)])
        self.Head = SpikingHead(self.config)
        
        

        
    def forward(self, x, targets=None, raw:bool=True):
        """
        

        Parameters
        ----------
        x : torch.Tensor
            The encoded input data in the form of a Tensor.
            Dimension should be Batch*Timestep*Sequence*Embedding
        targets : torch.tensor, optional
            The encoded target data in the form of a Tensor
            Dimension should be Batch*Timestep*Sequence*Embedding. Where
            the position encoding is skipped. The default is None.
        raw : bool
            Whether the input is of raw string format. In this case, the model
            will encode the input itself. The default is False.

        Returns
        -------
        preds : torch.tensor
            Prediction tensor of dimension classes, with prediction values 
            (not representative for probability distribution).
        Attention : torch.tensor
            The Value matrix from the self-attention block which represents
            the attention values between tokens.
        loss : torch.tensor
            The loss criterion computed for current input and target. Computed
            via cross entropy loss.

        """
        if(raw):
            if(type(x) == np.str_ or type(x) == str or (not (self.config.dictionary == None) and type(x[0]) == str)):
                x = [x]
            x = self.embedder.forward(x, fullSequence=False)
            if targets is not None:
                if(type(targets) == np.str_ or type(targets) == str or(not (self.config.dictionary == None) and type(x[0]) == str)):
                    targets = [targets]
                targets = self.embedder.forward(targets, target=True, fullSequence=False)

        T,B,S,E = x.shape
        
        self.Batch_size = B

        x = self.blocks(x)
        
        Attention = self.blocks[-1].Attention
        
        # Apply generation head
        preds = self.Head(x)
        
        # For training purposes: reset LIFs to prevent manipulation between
        # samples due to voltage levels, moved to generate and training loop
        # respectively
        functional.reset_net(self)
        
        if targets is not None:
            
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets.contiguous().view(-1))
            
            return preds, Attention, loss
        
        # We also return the explicit Value matrix from the Attention computation
        return preds, Attention, None

    def get_loss(self, x, targets):
        '''
        

        Parameters
        ----------
        x : torch.Tensor
            The encoded input tensor.
        targets : torch.Tensor
            The encoded target tensor.

        Returns
        -------
        loss : torch.Tensor
            The computed loss value.

        '''
        if(type(targets) == tuple):
            if(type(targets) == np.str_ or type(targets) == str):
                targets = [targets]
            targets = self.embedder.forward(targets, target=True)
        
        loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))

        return loss            

    
    @torch.no_grad()
    def generate(self, x: str, iterations: int = 1, create_visuals_up_to : int = 0,
                 paths : list = None):
        """

        Parameters
        ----------
        x : str
            The raw input data in the form of a String.
        iterations: int, optional
            The number of autoregressive iterations the model
            should generate. The default is 1.
        create_visuals_up_to: int, 
             Whether to create prediction visuals up to a certain character index.
             The default is 0.
        paths : list, optional
             The list of paths to save the (two) visualizations at. The default is
             None.

        Returns
        -------
        The generated sequence of characters.

        """
        inputStr = x
        output = []
        fullOutput = ''
        flag = False
        reducedOutput = False
        recordedPreds = None
        counter = -1
        
        if(not self.config.dictionary == None):
            text = x.lower()
                #for word in re.split(r'(\s+)', line):
            text = text.replace('!', '.')
            text = text.replace('?', '.')
            text = text.replace(';', '')
            text = re.sub('([^a-zA-Z. \']+)', ' ', text)
            text = re.sub(r'(\d+\.\d+|\b[A-Z](?:\.[A-Z])*\b\.?)|([.,;:!?)])\s*', lambda x: x.group(1) or f'{x.group(2)} ', text)
            text = text.replace('. . .', '...')
            text = text.replace('.', ' .')
            text = text.replace("''", "'")
            x = re.sub(' +', ' ', text).split()

        for iteration in range(iterations):
            while(not flag): 
                counter = counter + 1
                # get the predictions
                preds, attention, loss = self.forward(x)
                
                if(len(x)<self.config.max_len):
                    idx_next = preds[:, len(x)-1, :]
                    reducedOutput = True
                else:
                    # focus on last timestep prediction for further computation
                    idx_next = preds[:, -1, :]
                if(counter<create_visuals_up_to):
                    if(not recordedPreds == None):
                        recordedPreds = torch.cat((recordedPreds, idx_next))
                    else:
                        recordedPreds = idx_next
                idx_next = torch.argmax(idx_next, dim=-1).flatten()
                
                nextChar = self.embedder.int_to_tok[idx_next.item()]
            
                # append sampled index to the running sequence
                if(not self.config.dictionary == None):
                    x = x + [nextChar]
                else:
                    x = x + nextChar
                output.append(nextChar)
                if(len(x) > self.config.max_len):
                    if(reducedOutput):
                        break
                    x = x[1:]
                if(nextChar == 'E'):
                    flag = True
                    break
                elif(len(output) == self.config.max_len):
                    flag = True
                    break
            output = []
            flag = False
            if(self.config.dictionary == None):
                fullOutput+="".join(x)
            else:
                fullOutput+=" ".join(x)

        if(not paths==None):
            self.embedder.showPrediction(recordedPreds[:len(recordedPreds)//2], path=paths[0])
            self.embedder.showPrediction(recordedPreds[len(recordedPreds)//2:], path=paths[1], indexOffset=len(recordedPreds)//2)
        return fullOutput
        
    def reset_LIFs(self):
        """
        A reset method to reinitialize the LIF neurons, as to prevent them from
        interfering with the backpropagation.
        
        Should be replaced by src.spikingjelly.clock_driven.functional.reset_net
        
        Akin to usage in Spikformer (trainer:680/768) and SpikeGPT (123)

        Returns
        -------
        None.

        """
        if(self.config.learning_MSLIF):
            self.SMHA.query_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            self.SMHA.key_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            self.SMHA.value_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            self.SMHA.att_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            self.SMHA.proj_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            self.MLP.LIF_1 = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            self.MLP.LIF_2 = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            self.Head.gen_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
        else:
            self.SMHA.query_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
            self.SMHA.key_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
            self.SMHA.value_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
            self.SMHA.att_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
            self.SMHA.proj_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
            self.MLP.LIF_1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
            self.MLP.LIF_2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
            self.Head.gen_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
        return None
    
    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        Separates parameters into those that will experience weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        Returned is the custom optimizer.
        Adapted from minGPT: https://github.com/karpathy/minGPT/tree/master
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm1d,
                                    MaskPowerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif(pn.endswith('v_threshold')):
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    
class SpikingDecoderBlock(nn.Module):
    """
    The SpikingDecoderBlock to stack in the decoder Model.
    
    Includes SMHA as well as MLP.
    """
    
    def __init__(self, embed_dim, config, num_heads, dim_hid, firstBlock : bool = False):
        '''
        

        Parameters
        ----------
        embed_dim : int, optional
            The number of elements representing one element from the vocabulary.
            Excluding the positional encoding. The default is 8.
        config: SDconfig
            The configuration object of the model. Optional, but if not passed, vocab_size
            and max_len have to be passed. .
        num_heads: int, optional
            The number of heads to be used in SSA. The default is 2..
        dim_hid : int, optional
            The hidden dimension factor in the MLP. The default is None.
        firstBlock : bool, optional
            Whether the current Block is the first one, which adjusts
            the first residual connection. The default is False.

        Returns
        -------
        The created SpikingDecoderBlock object.

        '''
        super().__init__()
        self.config = config
        self.firstBlock = firstBlock
        if(config.spiking_residuals>1):
            if(self.config.learning_MSLIF):
                self.SMHA_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=self.config.device)
            else:
                self.SMHA_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        self.SMHA = SpikingMultiHeadAttention(config.embed_dim, config, config.num_heads, attn_drop = config.attn_dropout, scale=config.attention_scale)
        if(firstBlock):
            if(self.config.learning_MSLIF):
                self.resLIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=self.config.device)
            else:
                self.resLIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        if(config.spiking_residuals>0):
            if(self.config.learning_MSLIF):
                self.blockLIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=self.config.device)
            else:
                self.blockLIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        self.MLP = MLP(config, config.embed_dim, dim_hid, dropout = config.mlp_dropout)
        
    def forward(self, x):
        '''
        

        Parameters
        ----------
        x : torch.Tensor
            The encoded input tensor.

        Returns
        -------
        x : torch.Tensor
            The generated Tensor.

        '''
        
        if(self.config.spiking_residuals>1):
            x = self.SMHA_LIF(x)
        X_attention, Attention = (self.SMHA(x))
        
        if(self.firstBlock):
            X_attention = self.resLIF(X_attention)
        
        # First residual connection around the spiking Multi-Head Attention Block
        x = x + X_attention
        if(self.config.spiking_residuals>0):
            x = self.blockLIF(x)
        # Second residual connection around the contents of the MLP Block
        x = x + self.MLP((x))
        self.Attention = Attention
        return x
        
            
    
class SpikingMultiHeadAttention(nn.Module):
    '''
    The Spiking Multi-head Attention module
    '''
    def __init__(self, embed_dim, config, num_heads=2, scale=0.125, attn_drop=0.):
        """
        

        Parameters
        ----------
        embed_dim : int
            The embedding dimension of the input (including the positional encoding).
        config : SDconfig
            The config object which contains the hyperparameter.
        num_heads : int, optional
            Number of heads for Multi-Head-Attention. The default is 2.
        scale : float, optional
            The scaling factor to control for big values in 
            Dot Product. The default is 0.125.
        attn_drop : float, optional
            The dropout to be applied on the attention matrix. The default is 0..

        Returns
        -------
        The created SMHA object.

        """
        super().__init__()
        assert embed_dim % num_heads == 0, f"The embedding dimension {embed_dim} should be divisible by num_heads {num_heads}."
        
        self.config = config
        
        if(config.spiking_residuals<=1):
            if(self.config.learning_MSLIF):
                self.input_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            else:
                self.input_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
        else:
            if(self.config.learning_MSLIF):
                self.output_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
            else:
                self.output_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=self.config.backend)
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = scale
        
        if(config.normalization == 'layer'):
            self.query_Norm = nn.LayerNorm(embed_dim)
            self.key_Norm = nn.LayerNorm(embed_dim)
            self.value_Norm = nn.LayerNorm(embed_dim)
            self.proj_Norm = nn.LayerNorm(embed_dim)
        elif(config.normalization == 'batch'):
            self.query_Norm = nn.Sequential(
                Transpose(),
                nn.BatchNorm1d(embed_dim),
                Transpose())
            self.key_Norm = nn.Sequential(
                Transpose(),
                nn.BatchNorm1d(embed_dim),
                Transpose())
            self.value_Norm = nn.Sequential(
                Transpose(),
                nn.BatchNorm1d(embed_dim),
                Transpose())
            self.proj_Norm = nn.Sequential(
                Transpose(),
                nn.BatchNorm1d(embed_dim),
                Transpose())
        else:
            self.query_Norm = MaskPowerNorm(embed_dim)
            self.key_Norm = MaskPowerNorm(embed_dim)
            self.value_Norm = MaskPowerNorm(embed_dim)
            self.proj_Norm = MaskPowerNorm(embed_dim)
        
        self.query_L = nn.Linear(embed_dim, embed_dim)
        if(self.config.learning_MSLIF):
            self.query_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
        else:
            self.query_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        
        self.key_L = nn.Linear(embed_dim, embed_dim)
        if(self.config.learning_MSLIF):
            self.key_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
        else:
            self.key_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        
        self.attn_dropout = nn.Dropout(attn_drop)
        
        self.value_L = nn.Linear(embed_dim, embed_dim)
        if(self.config.learning_MSLIF):
            self.value_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
        else:
            self.value_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        
        if(self.config.learning_MSLIF):
            self.att_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
        else:
            self.att_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        
        self.proj_L = nn.Linear(embed_dim, embed_dim)
        if(self.config.learning_MSLIF):
            self.proj_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=self.config.backend, device=self.config.device)
        else:
            self.proj_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        
        # create the corresponding matrix shape (filter out entries top right)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len))
            .unsqueeze(0).unsqueeze(0).unsqueeze(0)
        )
    
    
    def forward(self, x):
        """
        

        Parameters
        ----------
        x : torch.Tensor
            The processed input tensor.

        Returns
        -------
        Attention_Spikes : torch.Tensor
             The transformed input.
        Value : torch.Tensor
            The Value matrix of the KQV attention.

        """
        # Shape is assigned to Batch, Timestep, Sequence and Embedding Dimensions
        T,B,S,E = x.shape
        
        query_size = E//self.num_heads
        
        # Project potential float values to spike form, except if we use pure
        # spiking residuals
        if(self.config.spiking_residuals<=1):
            x = self.input_LIF(x)
        
        # Adjust shape for xqv by combining Batch and Timestep
        # helps with normalizing
        x_for_qkv = x.flatten(0,1)
        
        if(self.config.track_frequency):
            self.config.smha_in_nnz.append(x_for_qkv.count_nonzero().item()/T/B)
        
        # Apply first linear layer (learnable Query Matrix)
        Query = self.query_L(x_for_qkv)

        # Normalize output and reshape to origin dimensions 
        # Also make contiguous to allow effective accessing of neighboring
        # tensor entries
        Query = self.query_Norm(Query).reshape(T,B,S,E)
        
        # Apply leaky integrate and fire neuron, will not transform shape
        Query = self.query_LIF(Query)
        
        # Switch last two dimensions (Sequence and Embedding)
        # Reshape to include the head dimension in the end (Query size= Embedding/heads)
        # During this reshape, the embedding dimension might be reduced
        # Lastly switch Sequence and number of heads dimension
        # Make contiguous for effective access of neighbouring entries
        # Query shape is now (Batch, Timestep, Heads, Sequence, -1 (Embedding//Heads))
        Query = Query.reshape(T,B, S, self.num_heads, query_size).transpose(2,3).contiguous()
        
        if(self.config.track_frequency):
            self.config.query_nnz.append(Query.count_nonzero().item()/T/B)
        
        # Same process for Key
        Key = self.key_L(x_for_qkv)
        Key = self.key_Norm(Key).reshape(T,B,S,E)
        Key = self.key_LIF(Key)
        # Retierater over dimension transformation
        Key = Key.reshape(T, B, S, self.num_heads, query_size).transpose(2,3).contiguous()

        # Same process for Value
        Value = self.value_L(x_for_qkv)
        Value = self.value_Norm(Value).reshape(T,B,S,E)
        Value = self.value_LIF(Value)
        Value = Value.reshape(T, B, S, self.num_heads, query_size).transpose(2,3).contiguous()
        
        # Transpose Key Matrix for multiplication
        # Key shape is now (Batch, Timestep, Heads, -1 (Embedding//Heads), Sequence)
        Key_transposed = Key.transpose(-2,-1)
        
        # Apply the matrix multiplication between Query and Key^T
        Attention = torch.matmul(Query, Key_transposed)
        
        # Adjust the mask to the length of our sequence 
        Mask = self.mask[:, :, :, :S, :S]
        
        # Apply mask (fill marked entries with 0)
        Attention = Attention.masked_fill(Mask == 0, 0)
        
        # Optional apply attention dropout
        if(self.attn_dropout != 0.):
            Attention = self.attn_dropout(Attention)
            
        if(self.config.track_frequency):
            self.config.att_nnz.append(Attention.count_nonzero().item()/T/B)
        
        # Multiply the Attention scores with Value matrix
        Attention = torch.matmul(Attention, Value)
        
        # Scale the Attention scores to control for big values
        # Attention now has shape (Batch, Timestep, Heads, Sequence, -1 (heads/embedding))
        Attention = Attention * self.scale
        
        # Bring num_head dimension to position 3 and combine together with 
        # head contents to receive embedding vector again
        Attention = Attention.transpose(2, 3).reshape(T, B, S, E).contiguous()
        
        # Apply leaky integrate and fire on attention matrix
        Attention_Spikes = self.att_LIF(Attention)
        
        if(self.config.track_frequency):
            self.config.smha_out_nnz.append(Attention_Spikes.count_nonzero().item()/T/B)
        
        # Collapse Timestep and Batch dimension for next leaky integrate and 
        # fire neuron
        Attention_Spikes = Attention_Spikes.flatten(0,1)
        
        #Attention_Spikes = self.proj_LN(Attention_Spikes)
        
        Attention_Spikes = self.proj_L(Attention_Spikes)
        
        Attention_Spikes = self.proj_Norm(Attention_Spikes).reshape(T,B,S,E)
        
        # Removed in Spikingformer
        if(self.config.spiking_residuals>1):
            Attention_Spikes = self.output_LIF(Attention_Spikes)
        
        return Attention_Spikes, Value
    
class MLP(nn.Module):
       '''
       The MLP which processes the output of SMHA
       '''
       def __init__(self, config, dim_in, dim_hid=None, dropout=0.):
            """
            

            Parameters
            ----------
            dim_in : int
                Input dimension size.
            dim_hid : int, optional
                Potential hidden dimension between layers. The default is None.
            dim_out : int, optional
                Output dimension size. The default is None.
            dropout : float, optional
                The dropout to be applied in the MLP. The default is 0..

            Returns
            -------
            None.

            """
            super().__init__()
            
            if(dim_hid):
                self.hidden = dim_in*dim_hid
            else:
                self.hidden = dim_hid or dim_in*4
                
            Flatter = nn.Flatten(0,1)
            
            Unflatter = nn.Unflatten(0, [config.timesteps, -1])
            
            self.config = config
            
            if(self.config.track_frequency):
                self.inC = setInCaller(self.config)
                self.outC = setOutCaller(self.config)
                
            if(config.normalization == 'layer'):
                Norm_1 = nn.LayerNorm(self.hidden)
                Norm_2 = nn.LayerNorm(dim_in)
            elif(config.normalization == 'batch'):
                Norm_1 = nn.Sequential(
                    Transpose(),
                    nn.BatchNorm1d(self.hidden),
                    Transpose())
                Norm_2 = nn.Sequential(
                    Transpose(),
                    nn.BatchNorm1d(dim_in),
                    Transpose())
            else:
                Norm_1 = MaskPowerNorm(self.hidden)
                Norm_2 = MaskPowerNorm(dim_in)
            
            if(dropout>0. and config.spiking_residuals == 0 and not config.learning_MSLIF):
                self.net = nn.Sequential(
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend),
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter,
                    nn.Dropout(dropout)
                )
            elif(dropout>0. and config.spiking_residuals == 0):
                self.net = nn.Sequential(
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device),
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter,
                    nn.Dropout(dropout)
                )
            elif(config.spiking_residuals == 0 and not config.learning_MSLIF):
                self.net = nn.Sequential(
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend),
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter
                )
            elif(config.spiking_residuals == 0):
                self.net = nn.Sequential(
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device),
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter
                )
            elif(dropout>0. and not config.learning_MSLIF):
                self.net = nn.Sequential(
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter,
                    nn.Dropout(dropout),
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
                )
            elif(dropout>0.):
                self.net = nn.Sequential(
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter,
                    nn.Dropout(dropout),
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device)
                )
            elif(not config.learning_MSLIF):
                self.net = nn.Sequential(
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter,
                    MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
                )
            else:
                self.net = nn.Sequential(
                    Flatter,
                    nn.Linear(dim_in, self.hidden),
                    Norm_1,
                    Unflatter,
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device),
                    Flatter,
                    nn.Linear(self.hidden, dim_in),
                    Norm_2,
                    Unflatter,
                    LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=config.device)
                )
            
            # Initialize hidden states at t=0 (for snntorch)
            # self.mem1 = self.LIF_1.init_leaky()
            # self.mem2 = self.LIF_2.init_leaky()
            
       def forward(self, x):
            """
            

            Parameters
            ----------
            x : torch.Tensor
                The tensor input to apply the MLP on.

            Returns
            -------
            x : torch.Tensor
                Output of the MLP. Dimension should be T,B,S,E

            """
            T,B,S,E = x.shape
            
            if(self.config.track_frequency):
                x = self.net[9](self.net[8](self.net[7](self.outC(self.net[6](self.net[5](self.net[4](self.net[3](self.net[2](self.inC(self.net[1](self.net[0](x)), T, B)))))), T, B))))
            else:
                x =  self.net(x)
            return x
        
class SpikingHead(nn.Module):
    '''
    The Spiking head object which transforms the processed value to output format.
    '''
    def __init__(self, config):
        """
        

        Parameters
        ----------
        config : SDconfig
            The configuration object for the model hyperparameters.

        Returns
        -------
        The created SDconfig object.

        """
        super().__init__()
        
        self.config = config
        
        if(config.spiking_head):
            if(self.config.learning_MSLIF):
                self.gen_LIF = LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=self.config.device)
            else:
                self.gen_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        
        if(config.spike_mode == 'concatenate'):
            self.gen_FC = nn.Linear(config.embed_dim*config.timesteps, config.vocab_size)
        else:
            self.gen_FC = nn.Linear(config.embed_dim, config.vocab_size)
        
    def forward(self,x):
        """
        

        Parameters
        ----------
        x : torch.Tensor
            The input of the spiking head layer.

        Returns
        -------
        TYPE
            The output of the spiking head. Should be a float-tensor with the
            corresponding values for each output class.

        """
        
        if(self.config.spiking_head and not self.config.float_reunite):
          x = self.gen_LIF(x)  
          
        if(self.config.spike_mode == 'average'):
            x = x.mean(0)
        elif(self.config.spike_mode == 'concatenate'):
            x = torch.cat([*[x[timestep] for timestep in range(x.size(0))]], dim=-1)
        elif(self.config.spike_mode == 'accumulate'):
            x = torch.sum(torch.stack([*[x[timestep] for timestep in range(x.size(0))]]), dim=0)
        elif(self.config.spike_mode == 'final'):
            # Only consider last timestep output
            x = x[-1, :, :, :]
        
        if(self.config.float_reunite):
            x = x.unsqueeze(0)
            x = self.gen_LIF(x)
            x = x.squeeze(0)
        if(self.config.track_frequency):
            self.config.head_nnz.append(x.count_nonzero().item())
        
        return self.gen_FC(x)
       #return self.gen_2nd_LIF(self.gen_FC(x))
       
class setInCaller(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x, T, B):
        self.config.mlp_in_nnz.append(x.count_nonzero().item()/T/B)
        return x
    
class setOutCaller(torch.nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, x, T, B):
        self.config.mlp_out_nnz.append(x.count_nonzero().item()/T/B)
        return x

    
class LinearNorm(nn.Module):
    '''
    The combined Layer Normalization and Linear layer.
    '''
    
    def __init__(self, dim_in, dim_out):
        """
        

        Parameters
        ----------
        dim_in : int
            Input dimension.
        dim_out : int
            Output dimension.

        Returns
        -------
        The created LinearNorm Object.

        """
        super().__init__()
        self.ff = nn.Linear(dim_in, dim_out)
        self.norm = nn.LayerNorm(dim_out)
    
    def forward(self,x):
        """
        

        Parameters
        ----------
        x : torch.Tensor
            The tensor to apply forward on.

        Returns
        -------
        TYPE torch.Tensor
            The output tensor.

        """
        return self.norm(self.ff(x))
    
class LearningMSLIF(MultiStepLIFNode):
    """
    Subclass of the Spikingjelly MultiStepLIFNode with the threshold 
    adjusted to a learnable parameter
    """
    def __init__(self, device : torch.device = torch.device('cpu'), *args, **kwargs):
        super().__init__(*args, **kwargs)
        v_threshold = self.v_threshold
        del self.v_threshold
        # set reference of threshold to point to torch Parameter
        self.v_threshold = nn.Parameter(torch.as_tensor(v_threshold))
        
class Transpose(torch.nn.Module):
   """
   Convenience module for inclusion in mlp sequential
   """
   
   def forward(self, x):
       return x.transpose(-1, -2)
