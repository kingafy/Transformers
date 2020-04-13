# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:00:51 2020

@author: anshu
"""

###STEP2

import tensorflow as tf

##step 1 to download the tar file 
import os
os.getcwd()
import tqdm
import re
import pandas as pd
os.getcwd()
os.chdir("C:/My_projects/BERT/Sentiment_analysis_imdb")

#provide datasets dir created in Step1 as part of tar unload
datasets_dir = "C:\My_projects\BERT\Sentiment_analysis_imdb\datasets"

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
  data = {}
  data["sentence"] = []
  data["sentiment"] = []
  
  
  ##for file_path in tqdm(os.listdir(directory), desc=os.path.basename(directory)):
  for file_path in os.listdir(directory):
    
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      data["sentence"].append(f.read())
      data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  print("I AM IN LOAD DATASET")
  print(os.path.join(directory, "pos"))
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

print(os.path.join(datasets_dir, "aclImdb", "train"))
train_df = load_dataset(os.path.join(datasets_dir, 
                                       "aclImdb", "train"))
test_df = load_dataset(os.path.join(datasets_dir, 
                                      "aclImdb", "test"))

##Final labelled data

train_df.to_csv("train_df.csv")
test_df.to_csv("test_df.csv")

directory = os.path.join(datasets_dir, 
                                       "aclImdb", "train")

