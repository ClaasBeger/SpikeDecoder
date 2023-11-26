# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 21:43:15 2023

@author: claas
"""

import pyperclip
import tkinter
from tkinter.messagebox import showerror
import ttkbootstrap as ttk
import pyttsx3
import model
import NonSpikingDecoderModel
import PartiallySpikingDecoderModel
import utils
import torch
from PIL import ImageTk, Image
from datetime import datetime
import re
import traceback
import numpy as np
import matplotlib.pyplot as plt

engine = pyttsx3.init()

class SpikingPredictor:
    
    def __init__(self, master):
        self.master = master
        # calling the UI method in the constructor
        self.MainWindow()
        # calling the Widgets method in the constructor
        self.Widgets()
        self.reader = utils.Reader()
        
        self.switch_to_words = False
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.SSconfig = model.SDconfig(30, 256, encodingType='learned', position_encoding_strategy='static', tok_embed_dim=50, heads=5, blocks=12, timesteps=4, device=device, normalization='PowerNorm', spike_mode="concatenate", track_frequency=True)
        self.SSdecoder = utils.load_model_from_checkpoint(model.SpikingDecoderModel, path_to_checkpoint='Pre-trained/StrictSpikes-10B30_DS900000_10 09 2023_12 30 57', config=self.SSconfig, dim_hid=16)
        #self.model = utils.load_model_from_checkpoint(model.SpikingDecoderModel, path_to_checkpoint='checkpoints/PeaceAndWar/checkpoint_epoch-150B64_DS90068_07 08 2023_20 08 51', config=config, dim_hid=16)

        #self.master.bind('<Enter>', self.predict)
        self.NSconfig = NonSpikingDecoderModel.NSDconfig(30, 256, embed_dim=80, heads=5, blocks=12, device=device)
        self.NSdecoder = utils.load_model_from_checkpoint(NonSpikingDecoderModel.NonSpikingDecoderModel, path_to_checkpoint='Pre-trained/NonSpikingDecoder_12Blocks-10B40_DS900000_09 09 2023_22 29 48', config=self.NSconfig, dim_hid=16)
        
        self.SDconfig = model.SDconfig(30, 256, encodingType='learned', position_encoding_strategy='static', tok_embed_dim=50, heads=1, blocks=12, timesteps=4, device=device, spike_mode='accumulate', learning_MSLIF=True, float_embedding=True, track_frequency=True)
        self.Sdecoder = utils.load_model_from_checkpoint(model.SpikingDecoderModel, path_to_checkpoint='Pre-trained/checkpoint_epoch-10B35_DS900000_12 09 2023_20 45 17', config=self.SDconfig, dim_hid=16)
        
        self.PSconfig = PartiallySpikingDecoderModel.PSDconfig(30, 256, encodingType='learned', position_encoding_strategy='static', tok_embed_dim=50, heads=5, blocks=12, timesteps=4, device=device, spike_degree=2, track_frequency=True)
        self.PSdecoder = utils.load_model_from_checkpoint(PartiallySpikingDecoderModel.PartiallySpikingDecoderModel, path_to_checkpoint='Pre-trained/PartiallySpiking-Decoder-10B42_DS900000_04 09 2023_05 12 02', config=self.PSconfig, dim_hid=16)
        
        self.WSDconfig = model.SDconfig(30, 128, encodingType='learned', position_encoding_strategy='static', tok_embed_dim=130, heads=8, blocks=25, timesteps=4, device=device, spike_mode='accumulate', learning_MSLIF=True, float_embedding=True, dictionary=self.reader.createDictionary(path='literature/Leo Tolstoy - War and Peace.txt', prune_level=6), track_frequency=True)
        self.WSDdecoder = utils.load_model_from_checkpoint(model.SpikingDecoderModel, path_to_checkpoint='Pre-trained/checkpoint_epoch-9B20_DS521263_22 11 2023_20 27 36', config=self.WSDconfig, dim_hid=16)
        
        self.WNSconfig = NonSpikingDecoderModel.NSDconfig(30, 128, embed_dim=160, heads=8, blocks=25, device=device, dictionary=self.reader.createDictionary(path='literature/Leo Tolstoy - War and Peace.txt', prune_level=6), track_frequency=True)
        self.WNSdecoder = utils.load_model_from_checkpoint(NonSpikingDecoderModel.NonSpikingDecoderModel, path_to_checkpoint='Pre-trained/checkpoint_epoch-3B20_DS521263_22 11 2023_22 28 17', config=self.WNSconfig, dim_hid=16)
        
        self.WPSconfig = PartiallySpikingDecoderModel.PSDconfig(30, 128, encodingType='learned', position_encoding_strategy='static', tok_embed_dim=130, heads=8, blocks=25, timesteps=4, device=device, spike_degree=2, dictionary=self.reader.createDictionary(path='literature/Leo Tolstoy - War and Peace.txt', prune_level=6), track_frequency=True)
        self.WPSdecoder = utils.load_model_from_checkpoint(PartiallySpikingDecoderModel.PartiallySpikingDecoderModel, path_to_checkpoint='Pre-trained/checkpoint_epoch-10B20_DS521263_24 11 2023_22 24 40', config=self.WPSconfig, dim_hid=16)
        
    def show_image(self, imagefile, second : bool = False):
        # Load the image
        image=Image.open(imagefile)

        # Resize the image in the given (width, height)
        img=image.resize((900,600))
        image = ImageTk.PhotoImage(img)
        if(second):
           self.predbox2.config(image=image)
           self.predbox2.image = image
           return
        self.predbox1.config(image=image)
        self.predbox1.image = image # save a reference of the image to avoid garbage collection
        
    def MainWindow(self):
        self.master.geometry('2000x1660+900+150')
        self.master.title('Spiking Predictor')
        self.master.resizable(width = 0, height = 0)
        
        icon = ttk.PhotoImage(file='icon.png')
        self.master.iconphoto(False, icon)
    
    def toggle_input_type(self):
        self.switch_to_words = not self.switch_to_words
        if not self.switch_to_words:
            self.toggle_btn.config(text="Switch to Word Input")
        else:
            self.toggle_btn.config(text="Switch to Char Input")
            
        def callback(*args):
            print("Variable changed")
        
    def Widgets(self):
        # the canvas for containing the other widgets
         self.canvas = ttk.Canvas(self.master, width = 2000, height = 1600)
         # the logo for the application
         self.logo = ttk.PhotoImage(file='icon.png').subsample(2, 2)
         self.canvas.create_image(1000, 95, image = self.logo)
         self.canvas.pack()
         # getting all the languages 
         models = ['Strict Spikes', 'SpikeDecoder', 'Performance Spikes', 'Non-Spiking']
         # first combobox for the source language
         self.pick_mode = ttk.Combobox(self.canvas, width = 20, bootstyle = 'primary', values = models)
         self.pick_mode.current(0)
         self.canvas.create_window(750, 275, window = self.pick_mode)
         
         self.toggle_btn = tkinter.Button(self.master, text="Switch to Words Input", command=self.toggle_input_type)
         self.canvas.create_window(1250, 275, window=self.toggle_btn)
         
         # loading the arrow icon
         # the second combobox for the destination language
         # scrollable text for entering input
         self.from_text = ttk.ScrolledText(self.master, font=("Dotum", 10), width = 30, height = 10)
         self.canvas.create_window(500, 600, window = self.from_text)
         
         # scrollable text for output
         self.to_text = ttk.ScrolledText(self.master, font=("Dotum", 10), width = 30, height = 10)
         self.canvas.create_window(1500, 600, window = self.to_text)
         
         self.predict_button = ttk.Button(self.master, text = 'Predict', width = 20, bootstyle = 'primary', command = self.predict)
         self.canvas.create_window(700, 950, window = self.predict_button)
         
         self.showEnergy_button = ttk.Button(self.master, text = 'Energy Consumption', width = 20, bootstyle = 'primary', command = self.showEnergyPlot)
         self.canvas.create_window(1300, 950, window = self.showEnergy_button)
         
         self.predbox1 = tkinter.Label(self.master)
         self.canvas.create_window(550, 1300, window = self.predbox1)
         
         self.predbox2 = tkinter.Label(self.master)
         self.canvas.create_window(1500, 1300, window = self.predbox2)
     
    def predict(self, *args):
        try:
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
            dt_string = re.sub(r'[^\w]', ' ', dt_string)
            # getting source language from first combobox via get() 
            self.model_mode = self.pick_mode.get()
            if(self.switch_to_words):
                if(self.model_mode == 'Non-Spiking'):
                    self.model = self.WNSdecoder
                elif(self.model_mode == "Performance Spikes"):
                    self.model = self.WSDdecoder
                else:
                    self.model = self.WSDdecoder
            elif(self.model_mode == 'Strict Spikes'):
                self.model = self.SSdecoder
            elif(self.model_mode == 'Non-Spiking'):
                self.model = self.NSdecoder
            elif(self.model_mode == 'Performance Spikes'):
                self.model = self.PSdecoder
            elif(self.model_mode == 'SpikeDecoder'):
                self.model = self.Sdecoder
    
            # getting every content fronm the first scrolledtext
            self.text = self.reader.clean(self.from_text.get(1.0, ttk.END))
            # translating the language
            self.prediction = self.model.generate(self.text, paths=['prediction_visualization/'+dt_string+'.png',
                                                                    'prediction_visualization/'+dt_string+'_2.png'],
                                                  create_visuals_up_to=32)
            
            self.show_image('prediction_visualization/'+dt_string+'.png')
            self.show_image('prediction_visualization/'+dt_string+'_2.png', second=True)

            
            
            # clearing the second scrolledtext
            self.to_text.delete(1.0, ttk.END)
            # inserting translation output in the second scroledtext
            self.to_text.insert(ttk.END, self.prediction)
            # activating the speak_button
            #self.speak_button.configure(state = ttk.ACTIVE)
            # activating the copy_button 
            #self.copy_button.configure(state = ttk.ACTIVE)
        # handle TypeError using except
        except TypeError as e:
            showerror(title='Invalid Input', message=e)
        # handle connection errors
        except Exception as e:
            traceback.print_exc()
            showerror(title='Exception', message=e)
    
    def showEnergyPlot(self, *args):
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
        dt_string = re.sub(r'[^\w]', ' ', dt_string)
        
        self.createEnergyPlot(path='prediction_visualization/'+dt_string+'_energy.png')
        
        self.show_image('prediction_visualization/'+dt_string+'_energy.png', second=True)

    def createEnergyPlot(self, path):
        energyValues = {'Strict Spikes' : self.computeConsumption('Strict Spikes', self.SSconfig.num_blocks), 
                        'Performance Spikes' : self.computeConsumption('Performance Spikes', self.PSconfig.num_blocks),
                        'Non-Spiking' : self.computeConsumption('Non-Spiking', self.NSconfig.num_blocks)}
        spike_type = list(energyValues.keys())
        consumption = list(energyValues.values())
        
        fig = plt.figure(figsize = (10, 5))
 
        # creating the bar plot
        plt.bar(spike_type, consumption, color ='maroon',
                width = 0.4)
 
        plt.xlabel("Spike Type")
        plt.ylabel(u"Energy consumption in \u03bcJ")
        plt.title("Energy comparision")
        
        plt.savefig(path)
        plt.clf()
        
        
    # TODO: Update computation with new method/spike frequency also find way to reset frequencies
    def computeConsumption(self, mode, blocks):
        if(mode=='Strict Spikes'):
            E = self.SSdecoder.config.embed_dim
            S = self.SSdecoder.config.max_len
            dim_hid = 16
            if(len(self.SSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.SSdecoder.config.query_nnz)/len(self.SSdecoder.config.query_nnz)
                Att_nnz = sum(self.SSdecoder.config.att_nnz)/len(self.SSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.SSdecoder.config.smha_in_nnz)/len(self.SSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.SSdecoder.config.smha_out_nnz)/len(self.SSdecoder.config.smha_out_nnz)
                MLP_in_nnz = sum(self.SSdecoder.config.mlp_in_nnz)/len(self.SSdecoder.config.mlp_in_nnz)
                MLP_out_nnz = sum(self.SSdecoder.config.mlp_out_nnz)/len(self.SSdecoder.config.mlp_out_nnz)
                Head_nnz = sum(self.SSdecoder.config.head_nnz)/len(self.SSdecoder.config.head_nnz)
            elif(len(self.PSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.PSdecoder.config.query_nnz)/len(self.PSdecoder.config.query_nnz)
                Att_nnz = sum(self.PSdecoder.config.att_nnz)/len(self.PSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.PSdecoder.config.smha_in_nnz)/len(self.PSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.PSdecoder.config.smha_out_nnz)/len(self.PSdecoder.config.smha_out_nnz)
                MLP_in_nnz = S*E*0.29
                MLP_out_nnz = S*E*dim_hid*0.11
                Head_nnz = S*E*0.3
            else:
                Query_nnz = S*E*0.3
                Att_nnz = S*E*0.16
                SMHA_in_nnz = S*E*0.27
                SMHA_out_nnz = S*E*0.23
                MLP_in_nnz = S*E*0.29
                MLP_out_nnz = S*E*dim_hid*0.11
                Head_nnz = S*E*0.4
            SSA = 2*Query_nnz*S+2*Att_nnz*E
            SMHA_in = 2*(SMHA_in_nnz)*E+S*E
            SMHA_out = 2*(SMHA_out_nnz)*E+S*E
            MLP_in = 2*(MLP_in_nnz)*dim_hid*E+S*E*dim_hid
            MLP_out = 2*(MLP_out_nnz)*E + E*S
            Head = 2*(Head_nnz)*self.SSdecoder.config.vocab_size +self.SSdecoder.config.vocab_size*S
            consumption = self.SSdecoder.config.timesteps*(blocks*(3*SMHA_in + SSA + SMHA_out + MLP_in + MLP_out)*0.9)
            consumption = (consumption+Head*0.9)/1000000
        elif(mode=='Word Strict Spikes'):
            E = self.WSDdecoder.config.embed_dim
            S = self.WSDdecoder.config.max_len
            dim_hid = 16
            if(len(self.WSDdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.WSDdecoder.config.query_nnz)/len(self.WSDdecoder.config.query_nnz)
                Att_nnz = sum(self.WSDdecoder.config.att_nnz)/len(self.WSDdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.WSDdecoder.config.smha_in_nnz)/len(self.WSDdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.WSDdecoder.config.smha_out_nnz)/len(self.WSDdecoder.config.smha_out_nnz)
                MLP_in_nnz = sum(self.WSDdecoder.config.mlp_in_nnz)/len(self.WSDdecoder.config.mlp_in_nnz)
                MLP_out_nnz = sum(self.WSDdecoder.config.mlp_out_nnz)/len(self.WSDdecoder.config.mlp_out_nnz)
                Head_nnz = sum(self.WSDdecoder.config.head_nnz)/len(self.WSDdecoder.config.head_nnz)
            elif(len(self.PSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.PSdecoder.config.query_nnz)/len(self.PSdecoder.config.query_nnz)
                Att_nnz = sum(self.PSdecoder.config.att_nnz)/len(self.PSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.PSdecoder.config.smha_in_nnz)/len(self.PSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.PSdecoder.config.smha_out_nnz)/len(self.PSdecoder.config.smha_out_nnz)
                MLP_in_nnz = S*E*0.29
                MLP_out_nnz = S*E*dim_hid*0.11
                Head_nnz = S*E*0.3
            else:
                Query_nnz = S*E*0.3
                Att_nnz = S*E*0.16
                SMHA_in_nnz = S*E*0.27
                SMHA_out_nnz = S*E*0.23
                MLP_in_nnz = S*E*0.29
                MLP_out_nnz = S*E*dim_hid*0.11
                Head_nnz = S*E*0.4
            SSA = 2*Query_nnz*S+2*Att_nnz*E
            SMHA_in = 2*(SMHA_in_nnz)*E+S*E
            SMHA_out = 2*(SMHA_out_nnz)*E+S*E
            MLP_in = 2*(MLP_in_nnz)*dim_hid*E+S*E*dim_hid
            MLP_out = 2*(MLP_out_nnz)*E + E*S
            Head = 2*(Head_nnz)*self.WSDdecoder.config.vocab_size +self.WSDdecoder.config.vocab_size*S
            consumption = self.WSDdecoder.config.timesteps*(blocks*(3*SMHA_in + SSA + SMHA_out + MLP_in + MLP_out)*0.9)
            consumption = (consumption+Head*0.9)/1000000
        elif(mode=='Non-Spiking'):
            E = self.NSdecoder.config.embed_dim
            S = self.NSdecoder.config.max_len
            dim_hid = 16
            consumption = blocks*(((2*E-1)*S**2+(2*S-1)*S*E+4*((2*E-1)*S*E+S*E)+((2*E-1)*S*(E*dim_hid)+E*S*dim_hid + (2*S*dim_hid)*S*E + S*E))*4.6)
            head = (2E-1)*S*self.NSdecoder.config.vocab_size + S*self.NSdecoder.config.vocab_size
            consumption = (consumption+head*4.6)/1000000
        elif(mode=='Word Non-Spiking'):
            E = self.WNSdecoder.config.embed_dim
            S = self.WNSdecoder.config.max_len
            dim_hid = 16
            consumption = blocks*(((2*E-1)*S**2+(2*S-1)*S*E+4*((2*E-1)*S*E+S*E)+((2*E-1)*S*(E*dim_hid)+E*S*dim_hid + (2*S*dim_hid)*S*E + S*E))*4.6)
            head = (2E-1)*S*self.WNSdecoder.config.vocab_size + S*self.WNSdecoder.config.vocab_size
            consumption = (consumption+head*4.6)/1000000
        elif(mode=='Performance Spikes'):
            E = self.PSdecoder.config.embed_dim
            S = self.PSdecoder.config.max_len
            if(len(self.PSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.PSdecoder.config.query_nnz)/len(self.PSdecoder.config.query_nnz)
                Att_nnz = sum(self.PSdecoder.config.att_nnz)/len(self.PSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.PSdecoder.config.smha_in_nnz)/len(self.PSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.PSdecoder.config.smha_out_nnz)/len(self.PSdecoder.config.smha_out_nnz)
            elif(len(self.SSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.SSdecoder.config.query_nnz)/len(self.SSdecoder.config.query_nnz)
                Att_nnz = sum(self.SSdecoder.config.att_nnz)/len(self.SSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.SSdecoder.config.smha_in_nnz)/len(self.SSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.SSdecoder.config.smha_out_nnz)/len(self.SSdecoder.config.smha_out_nnz)
            else:
                Query_nnz = S*E*0.3
                Att_nnz = S*E*0.16
                SMHA_in_nnz = S*E*0.27
                SMHA_out_nnz = S*E*0.23
            dim_hid = 16
            SSA = 2*Query_nnz*S+2*Att_nnz*E
            SMHA_in = 2*(SMHA_in_nnz)*E+S*E
            SMHA_out = 2*(SMHA_out_nnz)*E+S*E
            dim_hid = 16
            consumption = blocks*((SMHA_in+SSA+SMHA_out)*0.9*self.PSdecoder.config.timesteps+2*((2*E-1)*S*(E*dim_hid)+E*S*dim_hid)*4.6)
            head = (E-1)*S*self.PSdecoder.config.vocab_size + S*self.PSdecoder.config.vocab_size
            consumption = (consumption+head*0.9)/1000000
        elif(mode=='Word Performance Spikes'):
            E = self.WPSdecoder.config.embed_dim
            S = self.WPSdecoder.config.max_len
            if(len(self.WPSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.WPSdecoder.config.query_nnz)/len(self.WPSdecoder.config.query_nnz)
                Att_nnz = sum(self.WPSdecoder.config.att_nnz)/len(self.WPSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.WPSdecoder.config.smha_in_nnz)/len(self.WPSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.WPSdecoder.config.smha_out_nnz)/len(self.WPSdecoder.config.smha_out_nnz)
            elif(len(self.SSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.SSdecoder.config.query_nnz)/len(self.SSdecoder.config.query_nnz)
                Att_nnz = sum(self.SSdecoder.config.att_nnz)/len(self.SSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.SSdecoder.config.smha_in_nnz)/len(self.SSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.SSdecoder.config.smha_out_nnz)/len(self.SSdecoder.config.smha_out_nnz)
            else:
                Query_nnz = S*E*0.3
                Att_nnz = S*E*0.16
                SMHA_in_nnz = S*E*0.27
                SMHA_out_nnz = S*E*0.23
            dim_hid = 16
            SSA = 2*Query_nnz*S+2*Att_nnz*E
            SMHA_in = 2*(SMHA_in_nnz)*E+S*E
            SMHA_out = 2*(SMHA_out_nnz)*E+S*E
            dim_hid = 16
            consumption = blocks*((SMHA_in+SSA+SMHA_out)*0.9*self.PSdecoder.config.timesteps+2*((2*E-1)*S*(E*dim_hid)+E*S*dim_hid)*4.6)
            head = (E-1)*S*self.PSdecoder.config.vocab_size + S*self.PSdecoder.config.vocab_size
            consumption = (consumption+head*0.9)/1000000
        else:
            E = self.PSdecoder.config.embed_dim
            S = self.PSdecoder.config.max_len
            if(len(self.PSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.PSdecoder.config.query_nnz)/len(self.PSdecoder.config.query_nnz)
                Att_nnz = sum(self.PSdecoder.config.att_nnz)/len(self.PSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.PSdecoder.config.smha_in_nnz)/len(self.PSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.PSdecoder.config.smha_out_nnz)/len(self.PSdecoder.config.smha_out_nnz)
            elif(len(self.SSdecoder.config.query_nnz) != 0):
                Query_nnz = sum(self.SSdecoder.config.query_nnz)/len(self.SSdecoder.config.query_nnz)
                Att_nnz = sum(self.SSdecoder.config.att_nnz)/len(self.SSdecoder.config.att_nnz)
                SMHA_in_nnz = sum(self.SSdecoder.config.smha_in_nnz)/len(self.SSdecoder.config.smha_in_nnz)
                SMHA_out_nnz = sum(self.SSdecoder.config.smha_out_nnz)/len(self.SSdecoder.config.smha_out_nnz)
            else:
                Query_nnz = S*E*0.3
                Att_nnz = S*E*0.16
                SMHA_in_nnz = S*E*0.27
                SMHA_out_nnz = S*E*0.23
            dim_hid = 16
            SSA = 2*Query_nnz*S+2*Att_nnz*E
            SMHA_in = 2*(SMHA_in_nnz)*E+S*E
            SMHA_out = 2*(SMHA_out_nnz)*E+S*E
            dim_hid = 16
            consumption = blocks*((SMHA_in+SSA+SMHA_out)*0.9*self.PSdecoder.config.timesteps+2*((2*E-1)*S*(E*dim_hid)+E*S*dim_hid)*4.6)
            head = (E-1)*S*self.PSdecoder.config.vocab_size + S*self.PSdecoder.config.vocab_size
            consumption = (consumption+head*0.9)/1000000
            
        
        return consumption
    

    
root = ttk.Window(themename="cosmo")
application = SpikingPredictor(root)
root.bind_all("<Return>", lambda event: application.predict_button.invoke())
root.mainloop()