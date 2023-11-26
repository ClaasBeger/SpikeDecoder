import torch
from torch import Tensor
import os
from datetime import datetime
import re
import random
import math
import model#from SpikingTransformerDecoder 
import NonSpikingDecoderModel
from torch.utils.data import Dataset
from itertools import compress
import numpy as np
from spikingjelly.clock_driven import functional
from sklearn.utils import shuffle
from collections import defaultdict
#from gensim.utils import prune_vocab
import pickle


class Reader():
    '''
    The reader object which creates or reads in data.
    '''
    def __init__(self):
        '''
        
        Returns
        -------
        The creatd reader object.

        '''
        self.train_idx = 0
        self.val_idx = 0
        self.dictionary = None
        
    
    def read(self, path: str = 'literature/leo tolstoy - war and peace.txt', sampleLength: int = 10):
        """
        

        Parameters
        ----------
        path : str, optional
            The file path of the source.
            The default is 'literature/leo tolstoy - war and peace.txt'.
        sampleLength : int, optional
            The length of the generated samples. The default is 10.

        Returns
        -------
        data : list
            The generated samples.
        targets : list
            The corresponding sample targets.

        """
        words = list()
        with open(path, 'r') as f:
            text = ' '.join(f.read().split())
                #for word in re.split(r'(\s+)', line):
            text = text.replace('!', '.')
            text = text.replace('?', '.')
            text = text.replace(';', ' ')
            text = text.replace('--', ' ')
            text = re.sub('([^a-zA-Z. \']+)', '', text)
            text = fix_punctuation(text)
            full = text.lower()
        data = []
        targets = []
        cur = ''
        lastCur = ''
        target = False
        x = 0
        idx = 0
        newIdx = 0

        full = re.sub(' +', ' ', full)
        for x in range(len(full)):
            start = x
            if((start+sampleLength+1) > len(full)):
                break
            else:
                data.append((full[start:start+sampleLength]))
                targets.append(full[start+1:start+sampleLength+1])
        return data,targets
    
    def createDictionary(self, path: str = 'literature/leo tolstoy - war and peace.txt', prune_level=None,  srcLength:int = 0):
        """
        

        Parameters
        ----------
        path : str, optional
            Location of the source text file. The default is 'literature/leo tolstoy - war and peace.txt'.
        prune_level : int, optional
            Whether to prune words with less appearances than given level. The default is None.

        Returns
        -------
        words : list
            Sorted list of all words contained in the created dictionary.

        """
        file = open(path, 'r')
        text = file.read().lower()
        file.close()
        # replaces anything that is not a lowercase letter, a space, or an apostrophe with a space:
        text = re.sub('[^a-z\ \']+', " ", text)
        words = list(text.split())
        if(srcLength != 0):
            words = words[:srcLength]
        if(not prune_level == None):
            fq = defaultdict( int )
            for w in words:
                fq[w] += 1
            #prune_vocab(fq, prune_level)
            reducedWords = []
            for word in fq:
                if(not fq[word]<prune_level):
                  reducedWords = reducedWords + [word]
            words = reducedWords #fq.keys()
        words.append('.')
        words.append('P')
        words = sorted(set(words))
        self.dictionary = words
        return words
    
    def readWords(self, path: str = 'literature/leo tolstoy - war and peace.txt', sampleLength: int = 10):
        '''
        

        Parameters
        ----------
        path : str, optional
            The source path of the text. The default is 'literature/leo tolstoy - war and peace.txt'.
        sampleLength : int, optional
            The length of one text sample. The default is 10.

        Returns
        -------
        data : list
            The data array extracted from the source consisting of sampleLength words.
        targets : TYPE
            The target array corresponding to the data.

        '''
        with open(path, 'r') as f:
            text = f.read().lower()
                #for word in re.split(r'(\s+)', line):
            text = text.replace('!', '.')
            text = text.replace('?', '.')
            text = text.replace(';', '')
            text = re.sub('([^a-zA-Z. \']+)', ' ', text)
            text = fix_punctuation(text)
            text = text.replace('.', ' .')
            text = text.replace("''", "'")
        
        data = []
        targets = []
        cur = ''
        lastCur = ''
        target = False
        x = 0
        idx = 0
        newIdx = 0

        full = re.sub(' +', ' ', text).split()
        if(not self.dictionary == None):
            full = [word for word in full if word in self.dictionary]
        
        for x in range(len(full)):
            start = x
            if((start+sampleLength+1) > len(full)):
                break
            else:
                data.append((full[start:start+sampleLength]))
                targets.append(full[start+1:start+sampleLength+1])
        return data, targets
    
    def clean(self, x: str):
        """
        

        Parameters
        ----------
        x : str
            The string to be cleaned.

        Returns
        -------
        text : str
            The cleaned string.

        """
        text = ' '.join(x.split())
        text = text.replace('!', '.')
        text = text.replace('?', '.')
        text = text.replace(';', ' ')
        text = text.replace('--', ' ')
        text = re.sub('([^a-zA-Z. \']+)', '', text)
        text = fix_punctuation(text)
        text = text.lower()
        return text
        
        

    def unique(self, sequence):
        '''
        

        Parameters
        ----------
        sequence : iterable
            The sequence to extract the unique tokens from.

        Returns
        -------
        list
            The unique tokens.

        '''
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]

    
    def generateAlphabetSnippetsUpToLength(self, length: int, targetLength: int = None, shuffled: bool = True):
        '''
        

        Parameters
        ----------
        length : int
            The maximum length of Snippets to generate.
        targetLength : int, optional
            A fixed length of targets. The default is None.
        shuffled : bool, optional
            Whether to shuffle the created snippets array. The default is True.

        Returns
        -------
        snippets : list
            The list of created snippets.
        targets : list
            The list of created targets.

        '''
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        snippets = []
        targets = []
        for y in range(2, length+1):
            snipps, targs = self.generateAlphabetSnippetsOfLength(y, targetLength, shuffled)
            snippets = snippets +snipps
            targets = targets + targs

        singles, singlesTargets = self.generateAlphabetSnippetsOfLength(1, targetLength, shuffled)
        
        if(shuffled):
            shuffledSnippets, shuffledTargets = shuffle(snippets, targets)
        else:
            shuffledSnippets, shuffledTargets = snippets, targets
        snippets = singles + shuffledSnippets
        targets = singlesTargets + shuffledTargets
        
        return snippets, targets
    
    def generateAlphabetSnippetsOfLength(self, length: int, targetLength: int = None, shuffled: bool = True):
        '''
        

        Parameters
        ----------
        length : int
            The length of Snippets to generate.
        targetLength : int, optional
            A fixed length of targets. The default is None.
        shuffled : bool, optional
            Whether to shuffle the created snippets array. The default is True.

        Returns
        -------
        snippets : list
            The list of created snippets.
        targets : list
            The list of created targets.

        '''
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        snippets = []
        targets = []
        if(targetLength == None):
            targetLength = length
        for x in range(len(alphabet)):
            start = x
            if(start+length > len(alphabet)):
                snippets.append(
                    alphabet[start:] + alphabet[0:length-(len(alphabet) - start)])
                if((start+1)%(len(alphabet)) == 0):
                    targets.append(alphabet[0: length])
                    break
                targets.append(alphabet[(start+1)%len(alphabet):] + alphabet[((start+1) == len(alphabet)):length-(len(alphabet) - (start+1))])
            else:
                snippets.append((alphabet[start:start+length]))
                if((start+ length + 1) > len(alphabet)):
                    targets.append(alphabet[start+1: len(alphabet)] + alphabet[0])
                else:
                    targets.append(alphabet[start+1: (start+length+1)])
        
        if(shuffled):
            snippets, targets = shuffle(snippets, targets)

        return snippets, targets
    
    def generateRandomTextExcerpts(self, fullLength: int = 58, sampleLength: int = 5, seed:int = None):
        '''
        

        Parameters
        ----------
        fullLength : int, optional
            The length of the random text to generate (char count). The default is 58
        sampleLength : int, optional
            A fixed length of samples. The default is 5.

        Returns
        -------
        snippets : list
            The list of created snippets.
        targets : list
            The list of created targets.

        '''
        random.seed(seed)
        alphabet = "abcdefghijklmnopqrstuvwxyz' ."
        snippets = []
        targets = []
        full = ''
        curSnippet = ''
        lastSnippet = None
        for character in range(fullLength):
            full+=alphabet[random.randint(0, len(alphabet)-1)]
        for x in range(len(full)):
            start = x
            if((start+sampleLength+1) > len(full)):
                break
            else:
                snippets.append((full[start:start+sampleLength]))
                targets.append(full[start+1:start+sampleLength+1])
        return snippets, targets, full
    
class charDataset(Dataset):
    """
    A torch.utils.data.Dataset, which generates character samples from a given source.
    Depending on the split (train or test) the corresponding data subset will be accessed
    on __getitem__.
    """
    
    def __init__(self, src, split, inputLength: int = 6, fullLength: int = 130, seed: int = None, splitFactor: float = 0.9,
                 randomTest : bool = False, fullwords : bool = False):
        assert split in {'train', 'test'}
        assert src.startswith('literature/') or src in {'alphabet', 'random'}
        assert 0.<splitFactor<1. 
        self.split = split
        self.inputLength = inputLength
        self.fullLength = fullLength
        self.reader = Reader()
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz.\'SEP '
        if(fullwords):
            self.alphabet = self.reader.createDictionary(src)
        if(src == 'random'):
            self.data, self.targets, full = self.reader.generateRandomTextExcerpts(fullLength=fullLength, sampleLength=inputLength, seed=7)
        elif(src == 'alphabet'):
            self.data, self.targets, = self.reader.generateAlphabetSnippetsOfLength(inputLength)
        else:
            if(fullwords):
                self.data, self.targets = self.reader.readWords(src, sampleLength=inputLength)
            else:
                self.data, self.targets = self.reader.read(src, sampleLength=inputLength)
            if(not fullLength == 0):
                self.data = self.data[:fullLength]
                self.targets = self.targets[:fullLength]
        if(seed):
            np.random.seed(seed)
        if(not randomTest):
            mask_in = np.ones(round(len(self.data)*splitFactor))
            mask_out = np.zeros(len(self.data)-len(mask_in))
            self.mask = np.concatenate([mask_in, mask_out]) == 1
        else:
            self.mask = np.random.rand(len(self.data)) <= splitFactor  # first x% will be train, rest val
        if(split == 'train'):
            self.fdata = np.array(list(compress(self.data, self.mask)))
            self.ftargets = np.array(list(compress(self.targets, self.mask)))
        else:
            self.fdata = np.array(list(compress(self.data, ~self.mask)))
            self.ftargets = np.array(list(compress(self.targets, ~self.mask)))
            
    def __len__(self):
        return len(self.fdata)
    
    def get_vocab_size(self):
        return 30
    
    def get_block_size(self):
        return self.snipLength * 2 - 1
    
    def __getitem__(self, idx):

        x = self.fdata[idx]
        y = self.ftargets[idx]
        
        return x, y
    
class wordDataset(Dataset):
    """
    A torch.utils.data.Dataset, which generates word samples from a given source.
    Depending on the split (train or test) the corresponding data subset will be accessed
    on __getitem__.
    """
    
    def __init__(self, src, split, inputLength: int = 6, fullLength: int = 130, seed: int = None, splitFactor: float = 0.9,
                 randomTest : bool = False, reader : Reader = None):
        assert split in {'train', 'test'}
        assert src.startswith('literature/')
        assert 0.<splitFactor<1. 
        self.split = split
        self.inputLength = inputLength
        self.fullLength = fullLength
        if(not reader == None):
            self.reader = reader
            if(not self.reader.dictionary == None):
               self.alphabet = self.reader.dictionary
            else:
               self.alphabet = self.reader.createDictionary(src, srcLength=fullLength)
        else:
            self.reader = Reader()
            self.alphabet = self.reader.createDictionary(src, srcLength=fullLength)
        self.data, self.targets = self.reader.readWords(src, sampleLength=inputLength)
        if(not fullLength == 0):
                self.data = self.data[:fullLength]
                self.targets = self.targets[:fullLength]
        if(seed):
            np.random.seed(seed)
        if(not randomTest):
            mask_in = np.ones(round(len(self.data)*splitFactor))
            mask_out = np.zeros(len(self.data)-len(mask_in))
            self.mask = np.concatenate([mask_in, mask_out]) == 1
        else:
            self.mask = np.random.rand(len(self.data)) <= splitFactor  # first x% will be train, rest val
        if(split == 'train'):
            self.fdata = np.array(list(compress(self.data, self.mask)))
            self.ftargets = np.array(list(compress(self.targets, self.mask)))
        else:
            self.fdata = np.array(list(compress(self.data, ~self.mask)))
            self.ftargets = np.array(list(compress(self.targets, ~self.mask)))
            
    def __len__(self):
        return len(self.fdata)
    
    def get_vocab_size(self):
        return len(self.alphabet)
    
    def __getitem__(self, idx):

        x = self.fdata[idx]
        y = self.ftargets[idx]
        
        return x, y
    
def fix_punctuation(text: str):
    """
    

    Parameters
    ----------
    text : str
        The source text to correct.

    Returns
    -------
    text : str
        The corrected string.

    """
    #add space after punctuation
    text = re.sub(r'(\d+\.\d+|\b[A-Z](?:\.[A-Z])*\b\.?)|([.,;:!?)])\s*', lambda x: x.group(1) or f'{x.group(2)} ', text)
    # Add exception handling for ...
    text = text.replace('. . .', '...')
    return text
    
@torch.no_grad()
def estimate_loss(
    data: np.ndarray,
    targets: np.ndarray,
    model: torch.nn.Module,
    reader: Reader = None,
    embedding_Block: model.SpikingInputEmbeddingBlock = None,
    train: bool = True,
    eval_iters: int = 0,
    random : bool = False
):
    '''
    

    Parameters
    ----------
    data : iterable
        The data array.
    targets : iterable
        The targets array.
    model : torch.nn.Module
        The model to estimate the loss for.
    reader : Reader
        The reader object to use for sampling the batch.
    embedding_Block : model.SpikingInputEmbeddingBlock, optional
        The block to use for encoding inputs. The default is None.
    train : bool, optional
        Whether evaluation is done on training data. The default is True.
    eval_iters : int, optional
        The number of evaluation iterations. The default is 10.

    Returns
    -------
    out : float
        The mean of the computed losses.

    '''
    out = {}
    model.eval()
    losses = []
    idxs = np.arange(len(data))
    if(random):
        if(eval_iters>0):
            if(eval_iters > len(idxs)):
                eval_iters = len(idxs)
            draws = random.sample(set(idxs), eval_iters)
        else:
            draws = idxs
    else: 
        draws = [0]
    for idx in draws:
        if(random):
            X, Y = data[idx], targets[idx]
        else:
            X, Y = data, targets
        #X, Y = reader.get_batch(data=data, targets=targets, step_size = 1, train=train)
        if(embedding_Block):
            X = embedding_Block.encodeList(X, fullSequence=False)
            Y = embedding_Block.encodeList(Y, fullSequence=False, target=True)
        preds, attention, loss = model(X, Y, raw=True)
        if(len(loss.shape)>0 and loss.shape[0]>1):
            curlosses = [x.item() for x in loss]
            losses.extend(curlosses)
        else:
            losses.append(loss.item())
        functional.reset_net(model)
    model.train()
    out = np.array(losses).mean()
    return out


def load_model_from_checkpoint(
    model_class: torch.nn.Module,
    path_to_checkpoint: str = "checkpoints/state_dict_model.pt",
    **kwargs: dict,
) -> torch.nn.Module:
    '''
    

    Parameters
    ----------
    model_class : torch.nn.Module
        The class to load the model into.
    path_to_checkpoint : str, optional
        The path to the checkpoint. The default is "checkpoints/state_dict_model.pt".
    **kwargs : dict
        DESCRIPTION.

    Returns
    -------
    model : torch.nn.Module
        The loaded model.

    '''
    
    try:
        state_dict = torch.load(path_to_checkpoint+'state_dict.pt', map_location=torch.device('cpu'))
# =============================================================================
#         Insert for backwards compatibility
# 
        for entry in state_dict.copy().keys():
             if('proj_N.' in entry):
                 state_dict[entry.replace('proj_N.', 'proj_Norm.')] = state_dict.pop(entry)
             if(model_class == NonSpikingDecoderModel.NonSpikingDecoderModel):
                 if('.SMHA' in entry):
                     state_dict[entry.replace('.SMHA', '.NSMHA')] = state_dict.pop(entry)
                 if('.MLP' in entry):
                     state_dict[entry.replace('.MLP', '.NSMLP')] = state_dict.pop(entry)
# =============================================================================
        print("Successfully loaded model from the checkpoint")
    except Exception as e:
        print(f"Error loading the model from the checkpoint. {e}")
    model = model_class(**kwargs)
    # load the state_dict into the model
    #print(state_dict.keys())
    model.load_state_dict(state_dict)
    return model


def save_model_to_checkpoint(
    model: torch.nn.Module, batch_size: int, dataSize: int, path_to_checkpoint: str = "checkpoints", epoch: int = 0
):
    '''
    

    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    path_to_checkpoint : str, optional
        The path to save th emodel to. The default is "checkpoints".
    epoch : int, optional
        The number of training epochs. The default is 0.

    Returns
    -------
    None.

    '''
    # check if path exists, otherwise create it
    if not os.path.exists(path_to_checkpoint):
        os.makedirs(path_to_checkpoint)

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
    dt_string = re.sub(r'[^\w]', ' ', dt_string)
    checkpoint_name = "checkpoint_epoch-" + \
        str(epoch) + "B" + str(batch_size) + "_" + "DS" + str(dataSize) + "_" + dt_string + "_log.txt"
    checkpoint_name_dict = "checkpoint_epoch-" + \
        str(epoch) + "B" + str(batch_size) + "_" + "DS" + str(dataSize) +  "_" + dt_string + "state_dict" + ".pt"
    full_path = os.path.join(path_to_checkpoint, checkpoint_name)
    full_path_dict = os.path.join(path_to_checkpoint, checkpoint_name_dict)
    with open(full_path, 'w') as logfile:
        logfile.writelines(str(model.config))
    try:
        torch.save(model.state_dict(), full_path_dict)
        print("Successfully saved the model to {}".format(full_path_dict))
    except Exception as e:
        print(f"Error saving the model to checkpoint. {e}")
        
def save_dataset_to_text(dataset: np.array, path_to_dataset: str = "checkpoints", epoch: int = 0):
    '''
    

    Parameters
    ----------
    dataset : np.array
        The datasets to save.
    path_to_dataset : str, optional
        The path where to save the dataset. The default is "checkpoints".
    epoch : int, optional
        The number of training epochs. The default is 0.

    Returns
    -------
    None.

    '''
    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)

    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
    dt_string = re.sub(r'[^\w]', ' ', dt_string)
    dataset_name = "dataset_epoch-" + \
        str(epoch) + "_" + dt_string + ".pt"
    full_path = os.path.join(path_to_dataset, dataset_name)
    try:
        with open(full_path, 'wb') as file:
            pickle.dump(dataset, file)
        print("Successfully saved the model to {}".format(full_path))
    except Exception as e:
        print(f"Error saving the model to dataset. {e}")

def load_dataset_from_path(
    path_to_dataset: str,
    **kwargs: dict,
) -> torch.nn.Module:
    '''
    

    Parameters
    ----------
    path_to_dataset : str
        The path where the dataset was saved.
    **kwargs : dict
        DESCRIPTION.

    Returns
    -------
    train_data: list
        The training data.
    train_targets: list
        The training targets.
    val_data: list
        The validation data.
    val_targets: list
        The validation targets.
    full: str
        The full data string.

    '''
    
    try:
        with open(path_to_dataset, 'rb') as file:
            new_data = pickle.load(file)
        print("Successfully loaded dataset from the path")
    except Exception as e:
        print(f"Error loading the model from path. {e}")

    return new_data[0], new_data[1], new_data[2], new_data[3], new_data[4]

