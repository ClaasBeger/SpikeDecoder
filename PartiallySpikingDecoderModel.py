# -*- coding: utf-8 -*-
"""
Created on Thu May  4 04:24:14 2023

@author: claas
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import argmax
import math
import re
import model
import NonSpikingDecoderModel
from PowerNorm import MaskPowerNorm
if(torch.cuda.is_available()):
    import cupy

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function.
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
class PSDconfig:
    """
    The Partially spiking Decoder config class.
    Most of the Model specific hyperparameters are listed here
    """
    def __init__(
            self, vocab_size, max_len, spike_degree: int = 0, tok_embed_dim=30,
            pos_embed_dim = 30, attn_dropout = 0.0,
            mlp_dropout = 0.0, heads=2, blocks=1, device = torch.device('cpu'),
            encodingType='two-hot', position_encoding_strategy = 'linear',
            spiking_residuals: int = 0, spiking_head : bool = False, timesteps: int = 2,
            attention_scale : float = 0.125, SMHA_only : bool = False, normalization='layer',
            float_embedding : bool = False, combine_position : bool = False,
            firstBlock_spiking_res : bool = False, spike_mode : str = 'average',
            pos_bias : float = .0, float_reunite : bool = False, learning_MSLIF : bool = False,
            dictionary = None, track_frequency : bool = True, **kwargs
            ):
        """
        

        Parameters
        ----------
        vocab_size : int
            The size of the full vocabulary (including Start, End and Padding Token.)
        max_len : int
            Max length of the input sequence by character count (excl. Start and End Token).
        spike_degree : int
            The spike degree of the model, which describes the amount of spiking
            and non-spiking parts the model is created from.
        tok_embed_dim : int, optional
            The number of elements representing one element from the vocabulary.
            Excluding the positional encoding. Only needed for learned or otherwise
            variable encoding length. The default is 30.
        pos_embed_dim : int, optional
            The number of elements representing one positional index. 
            Only needed for learned or static strategy. The default is 30.
        attn_dropout : Float, optional
            The dropout to be applied on the attention matrix product. The default
            is 0.0.
        mlp_dropout : Float, optional
            The dropout to be applied inside of the mlp. The default is 0.0.
        heads : int, optional
            The number of heads to be used in SSA. The default is 2.
        blocks : int, optional
            The number of decoder blocks to generate for parallel processing.
        device : torch.device, optional
            The device to create the model on. The default is torch.device('cpu')
        encodingType : str, optional
            The representation strategy for the embedding. The default is 'two-hot'.
        position_encoding_strategy : str, optional
            The representation strategy for the position. The default is 'linear'.
        spiking_residuals : int, optional
            The degree to which the residuals should be made spiking. The default is 0.
        spiking_head : bool, optional
            Whether the classification head should operate on a spiking bases. 
            The default is True.
        timesteps : int, optional
            The number of timesteps for static repetition of the input. The 
            default is 2.
        attention_scale : Float, optional
            The number of steps to scale the Query-Key matrix by. The default is 0.125.
        SMHA_only : bool , optional
            Whether the model should operate on SMHA only basis, meaning it has 
            the same structure as a NSD, only with the SMHA inserted in the Partially
            Spiking Decoder Blocks (but still with float embedding). The default is False.
        normalization : str, optional
            The normalization layers to employ in the model. Possible choices are
            'batch', 'layer' and 'PowerNorm'. Layer is the fastest and produces the
            most stable results, but as opposed to the other two, it is not spike-compatible.
            The default is layer.
        float_embedding : bool, optional
            Whether to disable the mapping to binary value range of learned and static 
            embeddings, prior to computation inside the model. 
            The default value is False.
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
        learning_MSLIF : bool, optional
            Whether to use MultistepLIFs with learnable threshold. The default value is False.
        **kwargs : dict
            Potential additonal params.

        Returns
        -------
        None.

        """
        
        assert position_encoding_strategy in ['linear', 'binary', 'learned', 'static']
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.spike_degree = spike_degree
        self.attn_dropout = attn_dropout
        self.mlp_dropout = mlp_dropout
        self.num_blocks = blocks
        self.num_heads = heads
        self.position_encoding_strategy = position_encoding_strategy
        self.spiking_residuals = spiking_residuals
        self.spiking_head = spiking_head
        self.dictionary = dictionary
        if(dictionary != None):
            self.vocab_size = len(dictionary)
            vocab_size = len(dictionary)
        if(spike_degree==4):
            self.spiking_head = True
        self.timesteps = timesteps
        self.combine_position = combine_position
        self.tok_embed_dim = tok_embed_dim
        if(self.combine_position):
            self.pos_embed_dim = tok_embed_dim
        else:
            self.pos_embed_dim = pos_embed_dim
        self.attention_scale = attention_scale
        self.SMHA_only = SMHA_only
        self.normalization = normalization
        self.float_embedding = float_embedding
        self.firstBlock_spiking_res = firstBlock_spiking_res
        self.pos_bias = pos_bias
        self.spike_mode = spike_mode
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
        self.encodingType = encodingType
        if(device==torch.device('cpu') or self.learning_MSLIF):
            self.backend = 'torch'
        else:
            self.backend = 'cupy'
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __str__(self):
            return """PSD config with vocab size {vocab_size},
        max length {max_length}, 
        number of blocks {blocks}, 
        number of heads {heads},
        spike degree {spike_degree},
        position encoding strategy {pos_strategy},
        spiking residuals set to {spiking_residuals},
        spiking head set to {spiking_head},
        token embedding dimension set to {tok_embed_dim},
        position embedding dimension (if applicable) set to {pos_embed_dim},
        number of timesteps {timesteps},
        self attention scaling with {attention_scaling},
        SMHA only set to {SMHA_only},
        Combine Position set to {combine_position},
        float embedding set to {float_embedding},
        normalization of type {normalization},
        first Block spiking residual is set to {firstBlock_spiking_res},
        spike mode set to {spike_mode},
        static position encoding bias set to {pos_bias},
        float reunion set to {float_reunite},
        learning MSLIF set to {learning_MSLIF},
        and token embedding of type {tok_embedding}.""".format(vocab_size=self.vocab_size,
        max_length=self.max_len, blocks=self.num_blocks, heads=self.num_heads, spike_degree=self.spike_degree,
        pos_strategy=self.position_encoding_strategy, pos_embed_dim=self.pos_embed_dim,
        spiking_residuals=self.spiking_residuals, spiking_head=self.spiking_head, 
        tok_embed_dim=self.tok_embed_dim, timesteps=self.timesteps, attention_scaling = self.attention_scale,
        SMHA_only=self.SMHA_only, combine_position=self.combine_position, float_embedding=self.float_embedding, 
        normalization=self.normalization, firstBlock_spiking_res=self.firstBlock_spiking_res,
        tok_embedding=self.encodingType, pos_bias=self.pos_bias, float_reunite=self.float_reunite,
        learning_MSLIF=self.learning_MSLIF,spike_mode=self.spike_mode)


class PartiallySpikingDecoderModel(nn.Module):
    """
    The Spiking Decoder Model. 
    """
    
    def __init__(self, config : PSDconfig =None, vocab_size=None, max_len=None,
         spike_degree=0, dim_hid=None, dropout=0., embed_dim=30,
         heads=2, blocks=1, device = torch.device('cpu'), encodingType='two-hot',
         position_encoding_strategy = 'linear', learned_position=False, learning_MSLIF:bool=False):
        """
        
 
        Parameters
        ----------
        config: PSDconfig
            The configuration object of the model. Optional, but if not passed, vocab_size
            and max_len have to be passed. 
        vocab_size : int
            The size of the full vocabulary (including special tokens).
        max_len : int
            Max length of the input sequence by character count (excl. Start and End Token).
        encodingType : str
            The representation strategy for the embedding, either 'one-hot' or 'two-hot'.
        dim_hid : int, optional
            The hidden dimension in the MLP. The default is None.
        dropout : float, optional
            The dropout to be applied on the attention. The default is 0..
        embed_dim : int, optional
            The number of elements representing one element from the vocabulary.
            Excluding the positional encoding. The default is 8.
        heads: int, optional
            The number of heads to be used in SSA. The default is 2.
        blocks: int, optional
            The number of decoder blocks to generate for parallel processing.
            The default is 1.
        device: torch.device, optional
            The device to create the tensors on. Tje default is torch.device('cpu').
        position_encoding_strategy : str, optional
            The strategy to use for positional encoding. The default is 'linear'.
        learned_position : bool, optional
            Whether to use learned positional encoding for the non-spiking embedding block.
            The default is False.
        learning_MSLIF : bool, optional
            Whether to use MultistepLIFs with learnable threshold. The default value is False.

        Returns
        -------
        None.

        """
        super().__init__()
        
        assert (config != None or (vocab_size != None and max_len != None)), 'Either config or embedding dimension and max_len must be passed.'
        if(config != None):
           self.config = config 
        else:
            self.config = PSDconfig(vocab_size, max_len, spike_degree=spike_degree, embed_dim=embed_dim, dropout=dropout, heads=heads, blocks=blocks, device=device, encodingType=encodingType, position_encoding_strategy=position_encoding_strategy, learning_MSLIF=learning_MSLIF)
        if(self.config.SMHA_only):
            self.embedder = NonSpikingDecoderModel.NonSpikingInputEmbeddingBlock(self.config, learned_position)
            self.blocks = nn.Sequential(*[PartiallySpikingDecoderBlock(embed_dim, self.config, self.config.num_heads, dim_hid=dim_hid) for b in range(self.config.num_blocks)])
            self.Head = NonSpikingDecoderModel.NonSpikingHead(self.config)
            return
        if(self.config.spike_degree>0):
            self.embedder = model.SpikingInputEmbeddingBlock(self.config, create_timestep=(config.spike_degree>1))
        else:
            self.embedder = NonSpikingDecoderModel.NonSpikingInputEmbeddingBlock(self.config, learned_position)
        if(self.config.spike_degree>1):
            if(self.config.firstBlock_spiking_res):
                self.blocks = nn.Sequential(PartiallySpikingDecoderBlock(embed_dim, self.config, self.config.num_heads, dim_hid=dim_hid, firstBlock=True),*[PartiallySpikingDecoderBlock(embed_dim, self.config, self.config.num_heads, dim_hid=dim_hid) for b in range(self.config.num_blocks-1)])
            else:
                self.blocks = nn.Sequential(*[PartiallySpikingDecoderBlock(embed_dim, self.config, self.config.num_heads, dim_hid=dim_hid) for b in range(self.config.num_blocks)])
        else:
            self.blocks = nn.Sequential(*[NonSpikingDecoderModel.NonSpikingDecoderBlock(embed_dim, self.config, self.config.num_heads, attn_drop=self.config.attn_dropout, dim_hid=dim_hid, dropout=self.config.mlp_dropout) for b in range(self.config.num_blocks)])
        if(self.config.spike_degree <= 1):
            self.Head = NonSpikingDecoderModel.NonSpikingHead(self.config)
        else:
            self.Head = model.SpikingHead(self.config)
        
        
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
            Whether the input is in raw string format. The default is True.

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
            if(self.config.SMHA_only):
                x = x.repeat(self.config.timesteps, 1, 1, 1)
        
        if(self.config.spike_degree<=1 and not self.config.SMHA_only):
            B,S,E = x.shape
        else:
            T,B,S,E = x.shape
        
        x = self.blocks(x)
        
        Attention = self.blocks[-1].Attention
        
        preds = self.Head(x)
        
        if(self.config.spike_degree>1 or self.config.SMHA_only):
            functional.reset_net(self)
        
        if targets is not None:
            # Compute loss with the dimensions (B, C), where B denotes the Batch,
            # but actually represents Batch*Sequence, to compute the loss based
            # on an independent number of one character predictions
            # targets simply has shape B, filled with class indices
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), targets.contiguous().view(-1))
            
            return preds, Attention, loss
        
        # We also return the explicit Value matrix from the Attention computation
        return preds, Attention, None

    def get_loss(self, x, targets):
        targets = torch.reshape(targets, (self.Batch_size,-1))
        
        loss = F.cross_entropy(x, targets)

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
        embedder = NonSpikingInputEmbeddingBlock(self.config)
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
    
class PartiallySpikingDecoderBlock(nn.Module):
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
            Whether this is the first Block, which would adjust the first residual
            connection. The default is 0..

        Returns
        -------
        The created SpikingDecoderBlock object.

        '''
        super().__init__()
        self.config = config
        self.firstBlock = firstBlock
        if(firstBlock):
            if(config.learning_MSLIF):
                self.resLif = model.LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=self.config.device)
            else:
                self.resLIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        if(config.SMHA_only):
            self.SMHA = model.SpikingMultiHeadAttention(config.embed_dim, config, config.num_heads, attn_drop = config.attn_dropout)
            self.MLP = NonSpikingDecoderModel.NSMLP(self.config.embed_dim, dim_hid, self.config.mlp_dropout)
            return
        if(config.spiking_residuals>1):
            if(config.learning_MSLIF):
                self.SMHA_LIF = model.LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=self.config.device)
            else:
                self.SMHA_LIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
        self.SMHA = model.SpikingMultiHeadAttention(config.embed_dim, config, config.num_heads, attn_drop = config.attn_dropout)
        if(self.config.spike_degree>=3):
            if(config.spiking_residuals>0):
                if(config.learning_MSLIF):
                    self.blockLIF = model.LearningMSLIF(tau=2.0, detach_reset=True, backend=config.backend, device=self.config.device)
                else:
                    self.blockLIF = MultiStepLIFNode(tau=2.0, detach_reset=True, backend=config.backend)
            self.MLP = model.MLP(config, self.config.embed_dim, dim_hid, self.config.mlp_dropout)
        else:
            self.MLP = NonSpikingDecoderModel.NSMLP(config.embed_dim, dim_hid, dropout = config.mlp_dropout)
        
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
        if(self.config.spiking_residuals and self.config.spike_degree>2):
            x = self.blockLIF(x)
        # Second residual connection around the contents of the MLP Block
        x = x + self.MLP((x))
        self.Attention = Attention
        return x
    

