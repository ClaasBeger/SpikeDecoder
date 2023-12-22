# -*- coding: utf-8 -*-
"""
Created on Thu May  4 04:24:14 2023

@author: claas
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import argmax
import math
from PowerNorm import MaskPowerNorm
import re

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
class NSDconfig:
    """
    The spiking Decoder config class.
    Most of the Model specific hyperparameters are listed here
    """
    def __init__(
            self, vocab_size, max_len, embed_dim=8, dropout= 0.0,
            heads=2, blocks=1, device = torch.device('cpu'), normalization='layer',
            learned_embedding : bool = False, dictionary = None, **kwargs
            ):
        """
        

        Parameters
        ----------
        vocab_size : int
            The size of the full vocabulary (including Start, End and Padding Token.)
        max_len : int
            Max length of the input sequence by character count (excl. Start and End Token).
        encodingType : str
            The representation strategy for the embedding, either 'one-hot' or 'two-hot'.
        embed_dim : int, optional
            The number of elements representing one element from the vocabulary.
            Excluding the positional encoding. The default is 8.
        dropout : Float, optional
            The dropout to be applied on the attention matrix product. The default
            is 0.0.
        heads: int, optional
            The number of heads to be used in SSA. The default is 2.
        blocks : int, optional
            The number of decoder blocks to generate for parallel processing
        device : torch.device, optional
            The device to create the tensors on. The default is torch.device('cpu')
        normalization : str, optional
            The normalization type to incorporate. Should be 'layer', 'batch' or
            'PowerNorm'. The default is 'layer'.
        **kwargs : dict
            Potential additonal params.

        Returns
        -------
        None.

        """
        self.vocab_size = vocab_size
        # Adding two to account for Start and End Token
        self.max_len = max_len #+ 2
        self.attn_dropout = dropout
        self.num_blocks = blocks
        self.num_heads = heads
        self.embed_dim = embed_dim
        self.device = device
        self.learned_embedding = learned_embedding
        self.normalization = normalization
        self.dictionary = dictionary
        if(dictionary != None):
            self.vocab_size = len(dictionary)
            vocab_size = len(dictionary)
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    
    def __str__(self):
            return """NSD config with vocab size {vocab_size},
        max length {max_length}, 
        number of blocks {blocks}, 
        number of heads {heads},
        embedding dimension {emb_dimension},
        learned embedding set to {learned_embedding},
        and normalization of type {normalization}.""".format(vocab_size=self.vocab_size,
        max_length=self.max_len, blocks=self.num_blocks, heads=self.num_heads,
        emb_dimension=self.embed_dim, learned_embedding=self.learned_embedding, 
        normalization=self.normalization)

class NonSpikingInputEmbeddingBlock(nn.Module):
    """
    The Embedding Block to process and encode the input characters.
    """
    
    def __init__(self, config):
        """
        

        Parameters
        ----------
        config : NSDconfig
            The config object which denotes the model hyperparameters.

        Returns
        -------
        None.

        """
        super().__init__()
        #Create embedding dict
        self.config = config
        
        # define universe of possible input values
        # for older versions 
        # self.alphabet = 'abcdefghijklmnopqrstuvwxyzSEP.\' '
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz.\'SEP '
        if(self.config.vocab_size==30):
            self.alphabet = 'abcdefghijklmnopqrstuvwxyz.\'P '
        elif(self.config.dictionary != None):
            self.alphabet = self.config.dictionary
        
        self.token_embedding_table = nn.Embedding(self.config.vocab_size, self.config.embed_dim, device=self.config.device)
        if(self.config.learned_embedding):
            self.positional_encoding = nn.Embedding(self.config.vocab_size, self.config.embed_dim, device=self.config.device)
        else:
            self.positional_encoding = None
        
        # define a mapping of chars to integers
        self.tok_to_int = dict((c, i) for i, c in enumerate(self.alphabet))
        self.int_to_tok = dict((i, c) for i, c in enumerate(self.alphabet))
        
        
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

            
    def encode(self, InputSequence:str, target:bool=False, fullSequence:bool=False):
        """
        

        Parameters
        ----------
        InputSequence : str
            The sequence to be encoded.
        target : bool, optional
            Denotes whether this is part of encoding a target sequence.
            If so, positional encoding will be skipped. The default is False.
        fullSequence: bool, optional
            If the given sequence should be treated as a standalone (insert start
            and end tokens). The default is False
        Returns
        -------
        result : torch.Tensor
            The encoded input vector.

        """
        result = list()
        if(fullSequence):
            InputSequence = 'S' + InputSequence + 'E'
        # If neccessary, add padding to max len with zero vectors
        while(len(InputSequence)<self.config.max_len):
            if(self.config.dictionary==None):
                InputSequence = InputSequence + 'P'
            else:
                if(type(InputSequence) == np.ndarray):
                  InputSequence = np.append(InputSequence,'P')
                else:
                  InputSequence = InputSequence + ['P']
        if(target):
            integer_encoded = [self.tok_to_int[tok] for tok in InputSequence] 
            onehot_encoded = list()
            for value in integer_encoded:
               tensor = torch.as_tensor([value], device=self.config.device)
               onehot_encoded.append(tensor)
            result = torch.stack(onehot_encoded)
            result = result[:,0]
            # Add Batch dimension
            result = result.unsqueeze(0)
            return result
        encodedInput = torch.as_tensor([int(self.tok_to_int[tok]) for tok in InputSequence], device=self.config.device).int()
        result = self.token_embedding_table(encodedInput)
        result = result.unsqueeze(0)
        if(self.positional_encoding == None):
             result = result + self.position_encoding(self.config.max_len, self.config.embed_dim, device=self.config.device)
        else:
             result = result + self.positional_encoding(torch.arange(self.config.max_len, device=self.config.device))
        # Add Batch dimension
        #result = result.unsqueeze(0)
        return result
    
    def forward(self, InputSequence:list, target:bool=False, fullSequence:bool=False, training: bool = True):
        '''
        

        Parameters
        ----------
        InputSequence : list
            The list of given inputs representing one full batch.
        target : bool, optional
            Whether the given input sequence consists of targets (which would
            skip positional encoding). The default is False.
        fullSequence : bool, optional
            Wether the input consists of full Sequences (which would 
            include inserting start and end tokens). The default is False.
        training : bool, optional
            Whether the encoding happens in a training process. Will be considered
            in case of learned embeddings. The default is True.

        Returns
        -------
        torch.Tensor
             The encoded input elements.

        '''
        if(not training):
            self.eval()
        
        encodedTokens = []
        for token in InputSequence: 
            encodedTokens.append(self.encode(token, target, fullSequence))
        
        if(not training):
            self.train()
        
        return torch.cat(encodedTokens)
    
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
            The path to save the corresponding visualization under. The default is None.
        indexOffset : int, optional
            Whether to offset the displayed indexes by a certain value. The default is 0.
        top_k : int, optional
            The number of top choices by prediction to display in the visualization.
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
            axs = sns.heatmap(predictions, annot=[[tok for tok in self.alphabet]]*len(predictions), cmap = colormap, fmt = '',xticklabels=False, yticklabels=yticks, linewidth=1)
        
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
#             d[top+1] = [self.int_to_char[top_k[top].item()], pred[0][top_k[top].item()]]
#         print ("{:<8} {:<15} {:<10}".format('Pos','Char','Score'))
#         for k, v in d.items():
#             Char, Score = v
#             print ("{:<8} {:<15} {:<10}".format(k, Char, Score))
# =============================================================================

class NonSpikingDecoderModel(nn.Module):
    """
    The Spiking Decoder Model. 
    """
    
    def __init__(self, config : NSDconfig =None, vocab_size=None, max_len=None,
                 dim_hid=None, dim_out=None, dropout=0., embed_dim=35,
                 heads=2, blocks=1, learned_position=False):
        """
        

        Parameters
        ----------
        config: NSDconfig
            The configuration object of the model. Optional, but if not passed, vocab_size
            and max_len have to be passed. 
        vocab_size : int
            The size of the full vocabulary (including Start, End and Padding Token.
        max_len : int
            Max length of the input sequence by character count (excl. Start and End Token).
        dim_hid : int, optional
            The hidden dimension in the MLP. The default is None.
        dim_out : int, optional
            The out dimension in the MLP. The default is None.
        dropout : float, optional
            The dropout to be applied on the attention. The default is 0..
        embed_dim : int, optional
            The number of elements representing one element from the vocabulary.
            Excluding the positional encoding. The default is 8.
        heads: int, optional
            The number of heads to be used in SSA. The default is 2.
        blocks: int, optional
            The number of decoder blocks to generate for parallel processing
        learned_position : bool, optional
            Whether to use learned positional embeddings. If False, static will be used
            instead. The default is False.

        Returns
        -------
        None.

        """
        super().__init__()
        
        assert (config != None or (vocab_size != None and max_len != None)), 'Either config or embedding dimension and max_len must be passed.'
        if(config != None):
           self.config = config 
        else:
            self.config = NSDconfig(vocab_size, max_len, embed_dim=embed_dim, dropout=dropout, heads=heads, blocks=blocks)
        self.embedder = NonSpikingInputEmbeddingBlock(self.config)
        self.blocks = nn.Sequential(*[NonSpikingDecoderBlock(embed_dim, self.config, self.config.num_heads, dropout, dim_hid, dropout) for b in range(self.config.num_blocks)])
        self.Head = NonSpikingHead(self.config)
        
        
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
        raw : bool, optional
            Whether the input is in raw String form. The default is True.

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
            x = self.embedder(x, fullSequence=False)
            if targets is not None:
                if(type(targets) == np.str_ or type(targets) == str or(not (self.config.dictionary == None) and type(x[0]) == str)):
                    targets = [targets]
                targets = self.embedder(targets, target=True, fullSequence=False)
                
        B,S,E = x.shape

        x = self.blocks(x)
        
        Attention = self.blocks[-1].Attention
        
        # Apply classification head
        preds = self.Head(x)
        
        if targets is not None:
            
            # Compute loss with the dimensions (B, C), where B denotes the Batch,
            # but actually represents Batch*Sequence, to compute the loss based
            # on an independent number of one character predictions
            # targets simply has shape B, filled with class indices
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets.view(-1))
            
            return preds, Attention, loss
        
        # We also return the explicit Value matrix from the Attention computation
        return preds, Attention, None

    def get_loss(self, x, targets):
        '''
        

        Parameters
        ----------
        x : torch.Tensor
            The prediction tensor.
        targets : list
            The list of strings denoting the targets.

        Returns
        -------
        loss : torch.Tensor
            The loss value.

        '''
        if(type(targets) == np.str_ or type(targets) == str):
            targets = [targets]
        targets = self.embedder(targets, target=True, fullSequence=False)
        
        loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))

        return loss            

    @torch.no_grad()
    def generatePartialsWrapper(self, x: str, maxiter: int, prePreds : torch.Tensor, iterations: int = 1, create_visuals_up_to : int = 0,
                 paths : list = None):
        """

        Parameters
        ----------
        x : str
            
        iterations: int, optional
            
        create_visuals_up_to: int, optional
            
        paths : list, optional
        
        maxiter: int
            

        Returns
        -------
        The next generated token.
        """
        inputStr = x
        embedder = self.embedder
        output = []
        fullOutput = ''
        reducedOutput = False
        recordedPreds = prePreds
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
        
        if(not iterations==maxiter):
                # get the predictions
                preds, attention, loss = self.forward(x)
                
                if(len(x)<self.config.max_len):
                    idx_next = preds[:, len(x)-1, :]
                    reducedOutput = True
                else:
                    # focus on last timestep prediction for further computation
                    idx_next = preds[:, -1, :] 
                if(iterations<create_visuals_up_to):
                     if(not recordedPreds == None):
                         recordedPreds = torch.cat((recordedPreds, idx_next))
                     else:
                         recordedPreds = idx_next
                
                idx_next = torch.argmax(idx_next, dim=-1).flatten()
                
                nexttok = embedder.int_to_tok[idx_next.item()]
                
                # append sampled index to the running sequence
                if(not self.config.dictionary == None):
                    x = x + [nexttok]
                else:
                    x = x + nexttok
            
                # append sampled index to the running sequence
                if(self.config.dictionary == None):
                    fullOutput+="".join(x)
                else:
                    fullOutput+=" ".join(x)
        elif(not paths==None):
            self.embedder.showPrediction(recordedPreds[:len(recordedPreds)//2], path=paths[0])
            self.embedder.showPrediction(recordedPreds[len(recordedPreds)//2:], path=paths[1], indexOffset=len(recordedPreds)//2)
        return fullOutput, recordedPreds

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
        create_visuals_up_to: int, optional
            Whether to create visuals up to a certain index. The default is 0.
        paths : list, optional
            Whether to save created visuals under paths (visuals are split into
            two images).

        Returns
        -------
        The generated sequence of characters.

        """
        inputStr = x
        embedder = NonSpikingInputEmbeddingBlock(self.config)
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
                
                nexttok = embedder.int_to_tok[idx_next.item()]
                
                # append sampled index to the running sequence
                if(not self.config.dictionary == None):
                    x = x + [nexttok]
                else:
                    x = x + nexttok
            
                # append sampled index to the running sequence
                x = x + nexttok
                output.append(nexttok)
                if(len(x) > self.config.max_len):
                    if(reducedOutput):
                        break
                    x = x[1:]
                if(nexttok == 'E'):
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
    
    def configure_optimizer(self, weight_decay, learning_rate, betas):
        """
        Separates parameters into those that will experience weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        Returned is the custom optimizer.
        Adapted from minGPT: https://github.com/karpathy/minGPT/tree/master
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
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
    
class NonSpikingDecoderBlock(nn.Module):
    
    def __init__(self, embed_dim, config, num_heads, attn_drop, dim_hid, dropout):
        '''
        

        Parameters
        ----------
        embed_dim : int
            The embedding dimension.
        config : NSDconfig
            The model configuration object.
        num_heads : int
            The number of attention heads.
        attn_drop : float
            The dropout to be applied on the attention values.
        dim_hid : int
            The hidden dimension of the MLP.
        dropout : float
            The dropout to be applied in the MLP.

        Returns
        -------
        None.

        '''
        super().__init__()
        self.NSMHA = NonSpikingMultiHeadAttention(config.embed_dim, config, config.num_heads, attn_drop = config.attn_dropout)
        self.NSMLP = NSMLP(config.embed_dim, dim_hid, dropout)
        
    def forward(self, x, targets=None):
        X_attention, Attention = (self.NSMHA(x))
        
        # First residual connection around the spiking Multi-Head Attention Block
        x = x + X_attention
        
        # Second residual connection around the contents of the MLP Block
        x = x + self.NSMLP((x))
        self.Attention = Attention
        return x
        
            
    
class NonSpikingMultiHeadAttention(nn.Module):
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
        None.

        """
        super().__init__()
        assert embed_dim % num_heads == 0, f"The embedding dimension {embed_dim} should be divisible by num_heads {num_heads}."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = scale
        
        self.query_L = nn.Linear(embed_dim, embed_dim)
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

        self.key_L = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(attn_drop)
        
        self.value_L = nn.Linear(embed_dim, embed_dim)
        
        self.proj_L = nn.Linear(embed_dim, embed_dim)
        
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len))
            .unsqueeze(0).unsqueeze(0)
        )
    
    
    def forward(self, x):
        """
        

        Parameters
        ----------
        x : torch.Tensor
            The Tensor to apply the multi-head attention on.

        Returns
        -------
        Attention_Spikes : torch.Tensor
            The result of the multi-head attention object.
        Value : torch.Tensor
            The contents of the Value matrix.

        """
        # Shape is boxed into Batch, Sequence and Embedding Dimensions
        B,S,E = x.shape
        
        query_size = E//self.num_heads
        
        # Apply first linear layer (learnable Query Matrix)

        Query = self.query_L(x)
        Query = self.query_Norm(Query)
        # Switch last two dimensions (Sequence and Embedding)
        # Reshape to include the head dimension in the end (Query size= Embedding/heads)
        # During this reshape, the embedding dimension might be reduced
        # Lastly switch Sequence and number of heads dimension
        # Make contiguous for effective access of neighbouring entries
        # Query shape is now (Timestep, Batch, Heads, Sequence, -1 (Embedding//Heads))
        # Query.transpose(-1, -2)
        Query = Query.reshape(B, S, self.num_heads, query_size).transpose(1,2).contiguous()
        
        # Same process for Key
        Key = self.key_L(x)
        Key = self.key_Norm(Key)
        # Retierater over dimension transformation
        # Key.transpose(-1, -2)
        Key = Key.reshape(B, S, self.num_heads, query_size).transpose(1,2).contiguous()
        # Same process for Value
        Value = self.value_L(x)
        Value = self.value_Norm(Value)
        # Value.transpose(-1, -2)
        Value = Value.reshape(B, S, self.num_heads, query_size).transpose(1,2).contiguous()
        
        # Transpose Key Matrix for multiplication
        # Key shape is now (Timestep, Batch, Heads, -1 (Embedding//Heads), Sequence)
        Key_transposed = Key.transpose(-2,-1)
        
        # Apply the matrix multiplication between Query and Key^T
        Attention = torch.matmul(Query, Key_transposed) * (query_size**-0.5)
        
        # Adjust the mask to the length of our sequence 
        Mask = self.mask[:, :, :S, :S]
        
        # Apply mask (fill marked entries with 0)
        Attention = Attention.masked_fill(Mask == 0, float('-inf'))
        
        Attention = F.softmax(Attention, dim=-1)
        
        # Optional apply attention dropout
        if(self.attn_dropout != 0.):
            Attention = self.attn_dropout(Attention)
        
        # Multiply the Attention scores with Value matrix
        Attention = torch.matmul(Attention, Value)
        
        # Bring num_head dimension to position 3 and combine together with 
        # head contents to receive embedding vector again
        
        Attention = Attention.transpose(1, 2).contiguous().reshape(B, S, E)
        
        
        Attention = self.proj_L(Attention)
        
        Attention = self.proj_Norm(Attention)
        
        
        return Attention, Value
    
class NSMLP(nn.Module):
        def __init__(self, dim_in, dim_hid=None, dropout=0.):
            """
            

            Parameters
            ----------
            dim_in : int
                Input dimension size.
            dim_hid : int, optional
                Potential hidden dimension between layers. The default is None.
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
                self.hidden = dim_in*4
            if(dropout != None and dropout > 0.): 
                self.net = nn.Sequential(
                    nn.Linear(dim_in, self.hidden),
                    NewGELU(),
                    nn.Linear(self.hidden, dim_in),
                    nn.Dropout(dropout))
            else:
                self.net = nn.Sequential(
                    nn.Linear(dim_in, self.hidden),
                    NewGELU(),
                    nn.Linear(self.hidden, dim_in))
            
        def forward(self, x):
            """
            

            Parameters
            ----------
            x : torch.Tensor
                The tensor input to apply the MLP on.

            Returns
            -------
            x : torch.Tensor
                Output of the MLP. Dimension should be B,S,E

            """
            
            x = self.net(x)
            
            return x
        
class NonSpikingHead(nn.Module):
    def __init__(self, config):
        """
        

        Parameters
        ----------
        config : SDconfig
            The configuration object for the model hyperparameters.

        Returns
        -------
        None.

        """
        super().__init__()
    
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
        
        return self.gen_FC(x)
    
class LinearNorm(nn.Module):
    
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
        None.

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
        TYPE
            DESCRIPTION.

        """
        return self.norm(self.ff(x))
    
class Transpose(torch.nn.Module):
   """
   Convenience module for inclusion in mlp sequential
   """
   
   def forward(self, x):
       return x.transpose(-1, -2)
