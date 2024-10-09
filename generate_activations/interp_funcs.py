import re
import numpy as np
import spacy

def separate_words_commas_periods(sentence):
    
    words = sentence.split()
    words_puncs = []
    for w in words:
        if is_punctuation(w[-1]):
            words_puncs.append(w[:-1])
            words_puncs.append(w[-1])
        else:
            words_puncs.append(w)
        
    return words_puncs


def is_punctuation(char):
    # Define sentence-ending punctuation marks
    sentence_ending_punctuation = ['.', '!', '?', ',']
    return char in sentence_ending_punctuation

def is_sentence_ending_punctuation(char):
    # Define sentence-ending punctuation marks
    sentence_ending_punctuation = ['.', '!', '?']
    return char in sentence_ending_punctuation

def group_tokens(tokens_to_words_alignment):
    
    tokens_to_word_list = []
    prev_word_idx = 0
    token_idxs = []
    for t, w in enumerate(tokens_to_words_alignment):
        
        if w == prev_word_idx:
            token_idxs.append(t)
        else:
            tokens_to_word_list.append(token_idxs)
            token_idxs = [t]
            prev_word_idx = w
            
    tokens_to_word_list.append(token_idxs)
    return tokens_to_word_list