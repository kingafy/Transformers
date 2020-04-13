# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 13:49:42 2020

@author: anshu
"""


'''
Input IDs
The input ids are often the only required parameters to be passed to the model as input. They are token indices, numerical representations of tokens building the sequences that will be used as input by the model.
'''
##example of tokenizer in BERT

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

sequence = "A Titan RTX has 24GB of VRAM"

print(sequence)
print(tokenizer.tokenize(sequence))

#now converting to numbers
encoded_sequence = tokenizer.encode(sequence)
print(encoded_sequence)



'''
Attention mask
The attention mask is an optional argument used when batching sequences together. This argument indicates to the model which tokens should be attended to, and which should not.
'''
sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

encoded_sequence_a = tokenizer.encode(sequence_a)
assert len(encoded_sequence_a) == 8

encoded_sequence_b = tokenizer.encode(sequence_b)
assert len(encoded_sequence_b) == 19

'''
These two sequences have different lengths and therefore can’t be put together in a same tensor as-is. The first sequence needs to be padded up to the length of the second one, or the second one needs to be truncated down to the length of the first one.
'''
##pad the ones with smaller length
padded_sequence_a = tokenizer.encode(sequence_a, max_length=19, pad_to_max_length=True)

assert padded_sequence_a == [101, 1188, 1110, 170, 1603, 4954,  119, 102,    0,    0,    0,    0,    0,    0,    0,    0,   0,   0,   0]
assert encoded_sequence_b == [101, 1188, 1110, 170, 1897, 1263, 4954, 119, 1135, 1110, 1120, 1655, 2039, 1190, 1103, 4954, 138, 119, 102]


'''
Token Type IDs
Some models’ purpose is to do sequence classification or question answering. 
These require two different sequences to be encoded in the same input IDs. 
They are usually separated by special tokens, such as the classifier and separator tokens.
 For example, the BERT model builds its two sequence input as such:
    '''
sequence_a = "HuggingFace is based in NYC"
sequence_b = "Where is HuggingFace based?"

encoded_sequence = tokenizer.encode(sequence_a, sequence_b)
assert tokenizer.decode(encoded_sequence) == "[CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]"

'''However, other models such as BERT have an additional mechanism,
 which are the segment IDs. The Token Type IDs are a binary mask identifying the different sequences in the model.
 use encode_plus for those-->
 '''
 
 
 # Continuation of the previous script
encoded_dict = tokenizer.encode_plus(sequence_a, sequence_b)

assert encoded_dict['input_ids'] == [101, 20164, 10932, 2271, 7954, 1110, 1359, 1107, 17520, 102, 2777, 1110, 20164, 10932, 2271, 7954, 1359, 136, 102]
assert encoded_dict['token_type_ids'] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

print(tokenizer.tokenize(sequence_a))
print(tokenizer.encode_plus(sequence_a))
'''
{'input_ids': [101, 20164, 10932, 2271, 7954, 1110, 1359, 1107, 17520, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
'''




# Continuation of the previous script
encoded_dict = tokenizer.encode_plus(sequence_a, sequence_b)
print(encoded_dict)

assert encoded_dict['input_ids'] == [101, 20164, 10932, 2271, 7954, 1110, 1359, 1107, 17520, 102, 2777, 1110, 20164, 10932, 2271, 7954, 1359, 136, 102]
assert encoded_dict['token_type_ids'] == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]


'''
Position IDs
The position IDs are used by the model to identify which token is at which position. Contrary to RNNs that have the position of each token embedded within them, transformers are unaware of the position of each token. The position IDs are created for this purpose.

They are an optional parameter. If no position IDs are passed to the model, they are automatically created as absolute positional embeddings.
'''