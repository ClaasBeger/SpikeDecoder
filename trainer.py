import torch
import model
import math
import numpy as np
import NonSpikingDecoderModel
import PartiallySpikingDecoderModel

from utils import (
    save_model_to_checkpoint,
    estimate_loss,
    load_model_from_checkpoint,
    save_dataset_to_text,
    load_dataset_from_path,
    Reader,
    charDataset,
    wordDataset
)
from itertools import compress

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.dataloader import DataLoader

from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from spikingjelly.clock_driven import functional

from torch.profiler import profile, record_function, ProfilerActivity

import Visualizations

import time
start_time = time.time()


def get_num_correct(preds, labels, NonSpiking: bool = False):
    '''
    

    Parameters
    ----------
    preds : torch.Tensor
        The predictions made by the model.
    labels : torch.Tensor
        The labels for the samples the model made the predictions on.
    NonSpiking : bool, optional
        Whether the model that made the predictions is NonSpiking. The default is False.

    Returns
    -------
    int
        The number of correct samples.

    '''
    
    idxs_next = torch.argmax(preds, dim=-1)
    
    return idxs_next.eq(labels).sum().item()

def get_accuracy(m, data):
    hits=0
    count=0
    for sample in range(len(data.fdata)):
        x = m(data.fdata[sample])[0]
        targets = decoder.embedder([data.ftargets[sample]], target=True)
        hits+=get_num_correct(x, targets)
        count+=(len(data.fdata[sample]))
    print(hits/count)

def split_batch(batch, devices):
    """
    Split given batch among devices

    Parameters
    ----------
    batch : torch.Tensor
        One training batch.
    devices : list
        The list of devices to distribute the batch over.

    Returns
    -------
    split_batch : list
        List of the split elements from the batch.

    """
    batch_size = batch.size(0)
    split_size = batch_size // len(devices)
    splits = [split_size] * (len(devices) - 1)
    splits.append(batch_size - sum(splits))
    offset = 0
    split_batch = []
    for i, device in enumerate(devices):
        split = batch[offset:offset + splits[i]].to(device)
        split_batch.append(split)
        offset += splits[i]
    return split_batch

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
    print(output)
    # comment out for now, generated file is too big
    #p.export_chrome_trace("../trace_" + str(p.step_num) + ".json")
    
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

class Trainer:
    """
    The trainer object class to administer the model optimization.
    """
    
    def __init__(self, config, model, train_dataset: torch.utils.data.Dataset,
                 test_dataset: torch.utils.data.Dataset, epochs: int, 
                 documentation: bool = False, lr: float = 0.001, batch_size: int = 64,
                 annealing: bool = True, inputLength: int = 128, gradientClipping: bool = True,
                 checkPointRate : int = 100, save_dataset: bool = False, save_model: bool = True,
                 num_workers: int = 0, prevEpochs: int = 0, profiling: bool = False
                 ):
        """
        

        Parameters
        ----------
        config : object
            The configuration object for the respective model. Can be in the form
            NSDconfig, PSDconfig or SDconfig.
        model : object
            The model to be trained. Can be in the form of model, NonSpikingDecoderModel
            or PartiallySpikingDecoderModel and must be of the same type as config.
        train_dataset : torch.utils.data.Dataset
            The dataset that generates the character string samples for the training.
        test_dataset : torch.utils.data.Dataset
            The dataset that generates the character string samples for the testing.
        epochs : int
            The number of epochs to train the model.
        documentation : bool, optional
            Whether to document loss and accuracy in Tensorboard. The default is False.
        lr : float, optional
            The learning rate to pass to the optimizer. The default is 0.001.
        batch_size : int, optional
            The batch size to accumulate loss and derive steps from. The default is 64.
        annealing : bool, optional
            Whether to use cosine annealing for the learning rate. The default is True.
        inputLength : int, optional
            The input length of the model. The default is 128.
        gradientClipping : bool, optional
            Whether to clip the gradients in training loop. The default is True.
        checkPointRate : int, optional
            The number of epochs after which to create checkpoints. The default is 100.
        save_dataset : bool, optional
            Whether to save the dataset after training has finished. The default is False.
        save_model : bool, optional
            Whether to save the model after training has finished. The default is True.
        num_workers : int, optional
            The number of workers to pass to the DataLoader as argument. The default is 0.
        prevEpochs : int, optional
            The number of previous epochs the model was already trained. The default is 0.
        profiling : bool, optional
            Whether to apply time profiling during training. The default is False.

        Returns
        -------
        The created trainer object.

        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.documentation = documentation
        self.LEARNING_RATE = lr
        self.BATCH_SIZE = batch_size
        self.annealing = annealing
        self.inputLength = inputLength
        self.gradientClipping = gradientClipping
        self.checkPointRate = checkPointRate
        self.save_dataset = save_dataset
        self.save_model = save_model
        self.num_workers = num_workers
        self.prevEpochs = prevEpochs
        self.profiling = profiling
        
    def train(self):
        """
        The training method which administers the optimization process for the model.

        Returns
        -------
        None.

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if(self.documentation):
           writer = SummaryWriter() 
           
        NonSpiking = (type(config) == NonSpikingDecoderModel.NSDconfig or (type(config) == PartiallySpikingDecoderModel.PSDconfig and config.spike_degree<2)) 
        
        reader = Reader()


        # load model to GPU if available
        m = self.model.to(device)
        # print the number of parameters in the model
        print(
            "Model with {:.2f}M parameters".format(sum(p.numel() for p in m.parameters()) / 1e6)
        )
        # optimizer takes the model's parameters and the learning rate as input,
        # and updates the parameters during the training process in order to
        # minimize the loss function.

        optimizer = m.configure_optimizer(weight_decay=0.1, learning_rate=self.LEARNING_RATE, betas=(0.9,0.95))

        #scheduler = MultiStepLR(optimizer, 
        #                        milestones=[500, 1000, 1500], # List of epoch indices
        #                        gamma =0.5) # Multiplicative factor of learning rate decay

        scheduler = CosineAnnealingLR(optimizer, 200, eta_min=0.00005)
        
        if(self.documentation):
            EVAL_correct = 0

        train_loader = DataLoader(
               self.train_dataset,
               sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=False),
               shuffle=False,
               pin_memory=True,
               batch_size=self.BATCH_SIZE,
               num_workers=self.num_workers,
               collate_fn=my_collate
        )

        test_loader = DataLoader(
               self.test_dataset,
               sampler=torch.utils.data.RandomSampler(self.test_dataset, replacement=False),
               shuffle=False,
               pin_memory=True,
               batch_size=self.BATCH_SIZE,
               num_workers=self.num_workers,
               collate_fn=my_collate
        )

        EPOCH_SIZE = math.ceil(len(train_dataset)/self.BATCH_SIZE)

        train_iter = iter(train_loader)

        train_batch = next(train_iter)
        train_x, train_y = train_batch

        test_iter = iter(test_loader)

        test_batch = next(test_iter)
        test_x, test_y = test_batch

        devices = []
        if torch.cuda.device_count() > 1:
            for idx in range(torch.cuda.device_count()):
                devices.append(torch.device("cuda:"+str(idx)))
            m = MyDataParallel(m, device_ids=devices)
            m = m.to(device)

        loss_train = estimate_loss(
                    data=train_x, targets=train_y, model=m, reader=reader, eval_iters=self.BATCH_SIZE# embedding_Block=embedder, 
                )

        loss_val = estimate_loss(
                    data=test_x, targets=test_y, model=m, reader=reader, train=False, eval_iters=self.BATCH_SIZE #embedding_Block=embedder, 
                )
        if(self.documentation):
            writer.add_scalar("Training Loss", loss_train, self.prevEpochs)
            writer.add_scalar("Validation Loss", loss_val, self.prevEpochs)
        print("epoch {:10} | train loss {:6.4f} | val loss {:6.4f}".format(self.prevEpochs, loss_train, loss_val))
        if(self.profiling):
            p = profile(
                 activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=torch.profiler.schedule(
                     wait=2,
                     warmup=1,
                     active=2),
                 on_trace_ready=trace_handler)
            p.start()

        for epoch in range (1,self.epochs+1):
            for batch_iter in range(EPOCH_SIZE):
                # sample a batch of data
                try:
                	batch = next(train_iter)
                except StopIteration:
                	train_iter = iter(train_loader)
                	batch = next(train_iter)

                x_train, y_train = batch
                
                logits, attention, loss = m.forward(x_train, y_train, raw=True)
                
                if(self.documentation):
                    y = m.embedder.forward(y_train, target=True, fullSequence=False, training=False)
                    EVAL_correct += get_num_correct(logits, y, NonSpiking=NonSpiking)
                if(len(devices)>1):
                    loss.mean().backward()
                else:
                    # backward() method on the loss variable calculates the gradients 
                    # of the loss with respect to the model's parameters.
                    loss.backward()
                # step() method on the optimizer updates the model's parameters 
                # using the calculated gradients, in order to minimize the loss.
                
                # Clip the gradients to norm float
                if(self.gradientClipping):
                   torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0) 
                
                # Only update weights every other 2 iterations
                # Effective batch size is doubled (Gradient accumulation)
                #if (step+1) % 2 == 0 or (step+1) == MAX_ITER:
                    # Update weights
                optimizer.step()        # Reset the gradients to None
                    # zero_grad() method sets the gradients of all parameters in the optimizer to zero
                optimizer.zero_grad(set_to_none=True)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            x_train, y_train = batch
            try:
                batch = next(test_iter)
            except StopIteration:
                test_iter = iter(test_loader)
                batch = next(test_iter)
            x_test, y_test = batch
            loss_train = estimate_loss(
                    data=x_train, targets=y_train, model=m, reader=reader, eval_iters=self.BATCH_SIZE*5# embedding_Block=embedder, 
                )
            
            loss_val = estimate_loss(
                    data=x_test, targets=y_test, model=m, reader=reader, train=False, eval_iters=self.BATCH_SIZE*5 #embedding_Block=embedder, 
                )
            
            if(self.documentation):
                writer.add_scalar("Training Loss", loss_train, self.prevEpochs+epoch)
                writer.add_scalar("Validation Loss", loss_val, self.prevEpochs+epoch)
                
            print("epoch {:10} | train loss {:6.4f} | val loss {:6.4f}".format(self.prevEpochs+epoch, loss_train, loss_val))
            if((not epoch==0) and self.documentation):
                writer.add_scalar("Accuracy", EVAL_correct/ (self.BATCH_SIZE*EPOCH_SIZE*self.config.max_len), self.prevEpochs+epoch)
    
            scheduler.step()
            if(self.profiling):
                p.step()
            if(epoch%self.checkPointRate == 0):
                path = "checkpoints/PeaceAndWar"
                if(type(self.config) == PartiallySpikingDecoderModel.PSDconfig):
                    path = "checkpoints/spike_degree_"+str(self.config.spike_degree)+'/PeaceAndWar'
                save_model_to_checkpoint(model=m, batch_size=self.BATCH_SIZE, dataSize=len(self.train_dataset.fdata),
                                         path_to_checkpoint = path, epoch=self.prevEpochs+epoch)
            EVAL_correct = 0
            
        if(self.profiling):
            p.stop()
        if(self.save_dataset):
            dataset = [self.train_dataset.fdata, self.test_dataset.fdata]
            path = "checkpoints/PeaceAndWar"
            if(type(self.config) == PartiallySpikingDecoderModel.PSDconfig):
                path = "checkpoints/spike_degree_"+str(self.config.spike_degree)+'/PeaceAndWar'
            save_dataset_to_text(dataset, path_to_dataset = path, epoch=self.prevEpochs+epoch)
        if(self.save_model):
            path = "checkpoints/PeaceAndWar"
            if(type(self.config) == PartiallySpikingDecoderModel.PSDconfig):
                path = "checkpoints/spike_degree_"+str(self.config.spike_degree)+'/PeaceAndWar'
            save_model_to_checkpoint(model=m, batch_size=self.BATCH_SIZE, dataSize=len(self.train_dataset.fdata),
                                     path_to_checkpoint = path, epoch=self.prevEpochs+epoch)
        if(self.documentation):
            writer.close()
        print("--- %s seconds ---" % (time.time() - start_time))
        
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reader = Reader()
    #config = model.SDconfig(30, 128, encodingType='learned', position_encoding_strategy='static', tok_embed_dim=130, heads=8, blocks=25, timesteps=4, device=device, spike_mode='accumulate', learning_MSLIF=False, float_embedding=True, normalization="PowerNorm", dictionary=reader.createDictionary(path='literature/text8.txt', prune_level=8, srcLength=1500000))
    #config = NonSpikingDecoderModel.NSDconfig(30, 128, embed_dim=160, heads=8, blocks=25, device=device, normalization='layer', dictionary=reader.createDictionary(path="literature/leo tolstoy - war and peace.txt", prune_level=6))
    config = PartiallySpikingDecoderModel.PSDconfig(30, 128, tok_embed_dim=130, encodingType='learned', position_encoding_strategy='static', heads=8, blocks=25, spike_degree=2, spiking_head=False, device=device, timesteps=4, float_embedding=True, dictionary=reader.createDictionary("literature/leo tolstoy - war and peace.txt", prune_level=6))
    
    print(str(config))
    
    # load model from checkpoint
    #decoder = load_model_from_checkpoint(model.SpikingDecoderModel, path_to_checkpoint='checkpoints/PeaceAndWar/checkpoint_epoch-6B20_DS521263_21 11 2023_19 35 14', config=config, dim_hid=16)
    #decoder = load_model_from_checkpoint(PartiallySpikingDecoderModel.PartiallySpikingDecoderModel, path_to_checkpoint='checkpoints/Thesis/SpikeTransition/Spike_Degree_4-1Head-50B38_DS90000_04 09 2023_06 20 44', config=config, dim_hid=16)
    
    prevEpochs = 0
    
    LEARNING_RATE = 0.0005
    
    BATCH_SIZE = 20
    NonSpiking = (type(config) == NonSpikingDecoderModel.NSDconfig or (type(config) == PartiallySpikingDecoderModel.PSDconfig and config.spike_degree<2))
    
    #decoder = model.SpikingDecoderModel(config, dim_hid=16)#model.SpikingDecoderModel(config, dim_hid=32)#NonSpikingDecoderModel.NonSpikingDecoderModel(config, dim_hid=64)#model.SpikingDecoderModel(config, dim_hid=32)#NonSpikingDecoderModel.NonSpikingDecoderModel(config, dim_hid=64, learned_position=True)
    #decoder = NonSpikingDecoderModel.NonSpikingDecoderModel(config, dim_hid=16)#model.SpikingDecoderModel(config, dim_hid=32)#NonSpikingDecoderModel.NonSpikingDecoderModel(config, dim_hid=64)#model.SpikingDecoderModel(config, dim_hid=32)#NonSpikingDecoderModel.NonSpikingDecoderModel(config, dim_hid=64, learned_position=True)
    decoder = PartiallySpikingDecoderModel.PartiallySpikingDecoderModel(config, dim_hid=16)
    
    # load model to GPU if available
    m = decoder.to(device)
    
    train_dataset = wordDataset('literature/leo tolstoy - war and peace.txt', 'train', inputLength=128, fullLength=0, seed=7070, reader = reader)
    test_dataset = wordDataset('literature/leo tolstoy - war and peace.txt', 'test', inputLength=128,  fullLength=0, seed=7070, reader=reader)
    
    
    trainer = Trainer(config, decoder, lr=LEARNING_RATE, train_dataset=train_dataset, test_dataset=test_dataset, batch_size=BATCH_SIZE, checkPointRate=3, documentation=False, epochs=5, save_model=True, num_workers=8, prevEpochs=prevEpochs)
    
    trainer.train()

