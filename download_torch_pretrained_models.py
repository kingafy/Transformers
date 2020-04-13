# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 22:45:58 2020

@author: anshu
"""
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
import os

import transformers
####for TF models
'''
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    TFAutoModel
)
'''
##for Torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel
    
)

import os
os.chdir("D:/My_projects/HUGGING_FACE_TORCH")

WORKING_DIR = Path("D:/My_projects/HUGGING_FACE_TORCH")

print('Transformers version',transformers.__version__) # Current version: 2.3.0

#def transformers_model_dowloader(pretrained_model_name_list = ['bert-base-uncased'], is_tf = True):
def transformers_model_dowloader(pretrained_model_name_list = ['bert-base-uncased'], is_tf = False):
    model_class = AutoModel
    
    ## for TF version download
    '''
    if is_tf:
        model_class = TFAutoModel
        
    '''
    for i, pretrained_model_name in enumerate(pretrained_model_name_list):
        print(i,'/',len(pretrained_model_name_list))
        print("Download model and tokenizer", pretrained_model_name)
        transformer_model = model_class.from_pretrained(pretrained_model_name)
        transformer_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
        NEW_DIR = WORKING_DIR / pretrained_model_name
    
        try:
            os.mkdir(NEW_DIR)
        except OSError:
            print("Creation of directory %s failed" % NEW_DIR)
        else:
            print ("Successfully created directory %s " % NEW_DIR)
    
        print("Save model and tokenizer", pretrained_model_name, 'in directory', NEW_DIR)
        transformer_model.save_pretrained(NEW_DIR)
        transformer_tokenizer.save_pretrained(NEW_DIR)
        
        
pretrained_model_name_list = ['bert-base-cased', 'bert-base-multilingual-cased', 'gpt2', 'distilroberta-base','distilbert-base-cased']
#pretrained_model_name_list = ['roberta-base']
transformers_model_dowloader(pretrained_model_name_list, is_tf = False)
