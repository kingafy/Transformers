# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 16:51:06 2020

@author: anshu
"""

#####STEP 1--TAR donwload process########
import tensorflow as tf
from tensorflow import keras
##step 1 to download the tar file 
import os
os.getcwd()

dataset = tf.keras.utils.get_file(
      fname="aclImdb.tar.gz", 
      origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
      extract=True,
      cache_dir=os.getcwd())

''' 
Final output is stored in datasets folder which is created by keras 
train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))

'''


  
  