# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:50:12 2020

@author: anshu
"""

##step 3 TOEKNIZATION USING BERT


import tensorflow as tf
from tensorflow import keras
import os
import re
import os
os.getcwd()
os.chdir("C:/My_projects/BERT/Sentiment_analysis_imdb")

import pandas as pd
import bert
model_dir = "C:/My_projects/BERT/Sentiment_analysis_imdb/bert_model/uncased_L-12_H-768_A-12"


from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights

#from bert.tokenization import FullTokenizer
from bert import bert_tokenization

from tqdm import tqdm
import numpy as np
'''
Let's use the MovieReviewData class below, to prepare/encode the data for feeding into our BERT model, by:

tokenizing the text
trim or pad it to a max_seq_len length
append the special tokens [CLS] and [SEP]
convert the string tokens to numerical IDs using the original model's token encoding from vocab.txt
'''

bert_ckpt_dir = "C:/My_projects/BERT/Sentiment_analysis_imdb/bert_model/uncased_L-12_H-768_A-12"
class MovieReviewData:
    DATA_COLUMN = "sentence"
    LABEL_COLUMN = "polarity"

    def __init__(self, tokenizer: bert_tokenization.FullTokenizer, sample_size=None, max_seq_len=1024):
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        self.max_seq_len = 0
        ##train, test = download_and_load_datasets()
        train = pd.read_csv("train_df.csv")
        test = pd.read_csv("test_df.csv")
        
        train, test = map(lambda df: df.reindex(df[MovieReviewData.DATA_COLUMN].str.len().sort_values().index), 
                          [train, test])
                
        if sample_size is not None:
            assert sample_size % 128 == 0
            train, test = train.head(sample_size), test.head(sample_size)
            # train, test = map(lambda df: df.sample(sample_size), [train, test])
        
        ((self.train_x, self.train_y),
         (self.test_x, self.test_y)) = map(self._prepare, [train, test])

        print("max seq_len", self.max_seq_len)
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        ((self.train_x, self.train_x_token_types),
         (self.test_x, self.test_x_token_types)) = map(self._pad, 
                                                       [self.train_x, self.test_x])

    def _prepare(self, df):
        x, y = [], []
        with tqdm(total=df.shape[0], unit_scale=True) as pbar:
            for ndx, row in df.iterrows():
                text, label = row[MovieReviewData.DATA_COLUMN], row[MovieReviewData.LABEL_COLUMN]
                tokens = self.tokenizer.tokenize(text)
                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                self.max_seq_len = max(self.max_seq_len, len(token_ids))
                x.append(token_ids)
                y.append(int(label))
                pbar.update()
        return np.array(x), np.array(y)

    def _pad(self, ids):
        x, t = [], []
        token_type_ids = [0] * self.max_seq_len
        for input_ids in ids:
            input_ids = input_ids[:min(len(input_ids), self.max_seq_len - 2)]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
            t.append(token_type_ids)
        return np.array(x), np.array(t)
    
tokenizer = bert_tokenization.FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))
data = MovieReviewData(tokenizer, 
                       sample_size=10*128*2,#5000, 
                       max_seq_len=128)


##EDA for training data

print("            train_x", data.train_x.shape)
print("train_x_token_types", data.train_x_token_types.shape)
print("            train_y", data.train_y.shape)

print("             test_x", data.test_x.shape)

print("        max_seq_len", data.max_seq_len)

print(type(data))

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu_devices[0], True)
import math
bert_ckpt_file   = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


def flatten_layers(root_layer):
    if isinstance(root_layer, keras.layers.Layer):
        yield root_layer
    for layer in root_layer._layers:
        for sub_layer in flatten_layers(layer):
            yield sub_layer


def freeze_bert_layers(l_bert):
    """
    Freezes all but LayerNorm and adapter layers - see arXiv:1902.00751.
    """
    for layer in flatten_layers(l_bert):
        if layer.name in ["LayerNorm", "adapter-down", "adapter-up"]:
            layer.trainable = True
        elif len(layer._layers) == 0:
            layer.trainable = False
        l_bert.embeddings_layer.trainable = False


def create_learning_rate_scheduler(max_learn_rate=5e-5,
                                   end_learn_rate=1e-7,
                                   warmup_epoch_count=10,
                                   total_epoch_count=90):

    def lr_scheduler(epoch):
        if epoch < warmup_epoch_count:
            res = (max_learn_rate/warmup_epoch_count) * (epoch + 1)
        else:
            res = max_learn_rate*math.exp(math.log(end_learn_rate/max_learn_rate)*(epoch-warmup_epoch_count+1)/(total_epoch_count-warmup_epoch_count+1))
        return float(res)
    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

    return learning_rate_scheduler

def create_model(max_seq_len, adapter_size=64):
  """Creates a classification model."""

  #adapter_size = 64  # see - arXiv:1902.00751

  # create the bert layer
  with tf.io.gfile.GFile(bert_config_file, "r") as reader:
      bc = StockBertConfig.from_json_string(reader.read())
      bert_params = map_stock_config_to_params(bc)
      bert_params.adapter_size = adapter_size
      bert = BertModelLayer.from_params(bert_params, name="bert")
        
  input_ids      = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
  # token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="token_type_ids")
  # output         = bert([input_ids, token_type_ids])
  output         = bert(input_ids)

  print("bert shape", output.shape)
  cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(output)
  cls_out = keras.layers.Dropout(0.5)(cls_out)
  logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
  logits = keras.layers.Dropout(0.5)(logits)
  logits = keras.layers.Dense(units=2, activation="softmax")(logits)

  # model = keras.Model(inputs=[input_ids, token_type_ids], outputs=logits)
  # model.build(input_shape=[(None, max_seq_len), (None, max_seq_len)])
  model = keras.Model(inputs=input_ids, outputs=logits)
  model.build(input_shape=(None, max_seq_len))

  # load the pre-trained model weights
  load_stock_weights(bert, bert_ckpt_file)

  # freeze weights if adapter-BERT is used
  if adapter_size is not None:
      freeze_bert_layers(bert)

  model.compile(optimizer=keras.optimizers.Adam(),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")])

  model.summary()
        
  return model


from datetime import datetime
adapter_size = None # use None to fine-tune all of BERT
model = create_model(data.max_seq_len, adapter_size=adapter_size)

##log_dir = "C:/My_projects/BERT/Sentiment_analysis_imdb/logs"
##  './ is current directory
#log_dir =  "logs_movie\\scalar\\" + datetime.utcnow().isoformat(sep=' ', timespec='milliseconds')

#log_dir= os.path.join("C:/My_projects/BERT/Sentiment_analysis_imdb/logs",datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
log_dir= os.path.join("C:/My_projects/BERT/Sentiment_analysis_imdb/logs")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,profile_batch=100000000)

total_epoch_count = 10
# model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,
model.fit(x=data.train_x, y=data.train_y,
          validation_split=0.1,
          batch_size=12,
          shuffle=True,
          epochs=total_epoch_count,
          callbacks=[create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                    end_learn_rate=1e-7,
                                                    warmup_epoch_count=20,
                                                    total_epoch_count=total_epoch_count),
                     keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                     tensorboard_callback])



model.save_weights('./movie_reviews_epoch10.h5', overwrite=True)

'''
from datetime import datetime
print(datetime.utcnow().isoformat(sep=' ', timespec='milliseconds'))
'''