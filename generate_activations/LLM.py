import spacy
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import RobertaTokenizer, RobertaModel
import torch
import re
from spacy.tokenizer import Tokenizer
from spacy.training import Alignment
device_number = 2
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

import os
from interp_funcs import separate_words_commas_periods, group_tokens, is_punctuation
import argparse

'''
This script is used to generate activations from LLMs (GPT2-XL and RoBERTa)
for the Pereira, Fedorenko, and Blank datasets. 
'''

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--untrained", action='store_true', default=False, help="If true, generate activations for untrained model.")
parser.add_argument("--model", type=str, help="Model to generate activations for")
parser.add_argument("--model_num", type=str, default='', help="Seed number if specified")
parser.add_argument("--dataset", type=str, default='pereira', help="pereira, fedorenko, or blank")

args = parser.parse_args()
basePath = ''
savePath = '/data/LLMs/'
dataset = args.dataset
model_str = args.model
untrained = args.untrained
model_num = args.model_num
save_words = False
print("Untrained: ", untrained)
print("generating activations for: ", model_str)


basePath_data = '/data/LLMs/data_processed/'

# load linguistic stimuli 
if dataset == 'pereira':
    pereira_path = f"{basePath_data}{dataset}/text/sentences_ordered.txt"
    with open(pereira_path, "r") as file:
        # Read the contents line by line into a list
        experiment_txt = [line.strip() for line in file]
    # each element is exp-passage_name-passage_num-first half or second half
    data_labels = np.load(f"{basePath_data}{dataset}/dataset/data_labels_{dataset}.npy") 
    
if dataset == 'fedorenko':
    fed_path = f"{basePath}{dataset}_data/sentences_ordered.txt"
    with open(fed_path, "r") as file:
        # Read the contents line by line into a list
        experiment_txt = [line.strip() for line in file]
    data_labels = np.load(f"{basePath}data_processed/{dataset}/data_labels_{dataset}.npy")
    
if dataset == 'blank':
    blank_data = np.load(f"{basePath}{dataset}_data/story_data_dict.npz")
    experiment_txt = []
    data_labels = []
    for key, val in blank_data.items():
        experiment_txt.extend(val)
        data_labels.extend(np.repeat(key, len(val)))
    
    
# load models 
if 'gpt' in model_str:
    
    model = GPT2LMHeadModel.from_pretrained(model_str)
    
    tokenizer = GPT2Tokenizer.from_pretrained(f"{model_str}")
    
    if untrained:
        config = GPT2Config.from_pretrained(model_str)
        model = GPT2LMHeadModel(config)
        model_str += '-untrained'
        
    model.eval()
    model = model.to(device)  
    
    embedding_matrix = model.transformer.wte 
    positional_matrix = model.transformer.wpe
    
    
elif 'roberta' in model_str:
    
    tokenizer = RobertaTokenizer.from_pretrained(model_str)
    model = RobertaModel.from_pretrained(model_str)
    model.eval()
    model = model.to(device)    
    embedding_matrix = model.get_input_embeddings().weight.data
    positional_matrix = model.embeddings.position_embeddings.weight.data

    
def split_multipunc_tokens(toks):
    
    '''
    :param list toks: tokens from an LLM tokenizer
    
    The purpose of this function is to split tokens that are composed of only multiple punctuation marks.
    For instance " '. ". These tokens are problematic when trying to align tokens to words. 
    
    For instance, if we have the string: " 'car'. ", then the words list will contain ['car', '.'] (this is because
    the word lists separates periods and commas into their own elements, so not really a word list...). 
    If the last apostrophe and period are combined into a token, it is impossible to find an 
    alignment between tokens and words. 
    '''
    
    import string
    
    new_tokens = []
    
    for s in toks:
        
        if all(char in string.punctuation for char in s) and len(s) > 1:
            print("Splitting token: ", s)
            breakpoint()
            for char in s:
                new_tokens.append(char)
        else:
            new_tokens.append(s)
            
    return new_tokens

def pool_representations(dataset, contextual_embeddings, static_embeddings, 
                         static_embeddings_pos_only, static_embeddings_no_pos):
    
    '''
    :param str dataset: pereira, blank, or fedorenko
    :param ndarray contextual embeddings: contextual embeddings from an LLM,
    of shape num_tokens/num_words x num_layers x embed_size 
    :param ndarray static embeddings: static embeddings from an LLM, of shape
    num_tokens/num_words x embed_size
    :param ndarray static_embeddings_pos_only: positional embeddings
    :param ndarray static_embeddings_no_pos: static embeddings with no positional embeddings
    '''
    
    if dataset == 'pereira' or dataset == 'blank':
        
        activity_sent = contextual_embeddings[-1]
        activity_sent_sp = np.sum(contextual_embeddings, axis=0)
        static_activity_pos_embed = np.sum(static_embeddings, axis=0)
        static_activity_pos = np.sum(static_embeddings_pos_only, axis=0)
        static_activity_embed = np.sum(static_embeddings_no_pos, axis=0)
        
    elif dataset == 'fedorenko':
        
        activity_sent = contextual_embeddings
        activity_sent_sp = None
        static_activity_pos_embed = static_embeddings
        static_activity_pos = static_embeddings_pos_only
        static_activity_embed = static_embeddings_no_pos
            
    return activity_sent, activity_sent_sp, static_activity_pos_embed, static_activity_pos, static_activity_embed
        
    

def get_model_activity(previous_text, current_text, embedding_matrix, 
                               positional_matrix, tokenizer, model_str, dataset, save_words, 
                               max_context_size=512):
    
    '''
    :param str previous_text: linguistic stimuli which serves as previous context
    :param str current_text: the linguistic stimuli to obtain activations for
    :param torch tensor embedding_matrix: static embedding matrix
    :param torch tensor positional_matrix: static positional matrix
    :param tokenizer: tokenize sentence
    :param str model_str: model used to generate activations
    :param str dataset: neural dataset 
    :param bool save_words: if true, average across tokens for multi-token words
    :param int max_context_size: max context size for LLM (in tokens)
    '''
    
    # tokenize text 
    curr_tokens = tokenizer.tokenize(current_text)
    #curr_tokens = split_multipunc_tokens(curr_tokens)
    num_ct = len(curr_tokens)
    prev_tokens = tokenizer.tokenize(previous_text)
    tokens = prev_tokens + curr_tokens
    tokens = tokens[-max_context_size:]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # append start and end tokens for roberta
    if model_str == 'roberta-large':
        token_ids.insert(0, 0)
        token_ids.append(2)
        
    tensor_input = torch.tensor([token_ids])
    tensor_input = tensor_input.to(device)    
    
    with torch.no_grad():
        
        if 'gpt' in model_str:
            static_embed = embedding_matrix(tensor_input)
            static_pos = positional_matrix.weight[np.arange(len(tensor_input[0])), :].unsqueeze(0)   
            
        elif 'roberta' in model_str:
            static_embed  = embedding_matrix[tensor_input, :]
            static_pos = positional_matrix[np.arange(len(tensor_input[0])), :].unsqueeze(0)
            
        static_embed_pos = torch.squeeze(static_embed + static_pos) # ctx_size x embed_size 
        static_pos = torch.squeeze(static_pos)
        static_embed = torch.squeeze(static_embed)
        
        outputs = model(tensor_input, output_hidden_states=True, output_attentions=True)
        outputs = outputs.hidden_states
        # number of layers x context size x embedding size
        outputs = torch.stack(outputs).squeeze()
        
    # remove <s> and </s> tokens because we only want to sum across 
    # words/punctuation marks for bert-style models.
    if 'roberta' in model_str:
        static_embed_pos = static_embed_pos[1:-1]
        static_pos = static_pos[1:-1]
        static_embed = static_embed[1:-1]
        outputs = outputs[:, 1:-1, :]
        
    # only take tokens corresponding to the current text
    static_embed_pos = static_embed_pos[-num_ct:]
    static_pos = static_pos[-num_ct:]
    static_embed = static_embed[-num_ct:]
    outputs = outputs[:, -num_ct:]
    
    # in this case, we just pool across tokens 
    if save_words == False:
        
        activity_sent, activity_sent_sp, static_activity_pos_embed, static_activity_pos, static_activity_embed = \
        pool_representations(dataset, outputs.cpu().detach().numpy().swapaxes(0,1), static_embed_pos.cpu().detach().numpy(), 
        static_pos.cpu().detach().numpy(), static_embed.cpu().detach().numpy())
        
        non_averaged_static = static_embed_pos.cpu().detach().numpy()
    
    # average tokens for multi-token words (after separating periods and commas)
    else:
        
        tokens_curr_cleaned = [t.replace("Ġ", '') for t in curr_tokens] 
        
        words = separate_words_commas_periods(current_text)
        
        align = Alignment.from_strings(words, tokens_curr_cleaned)
        tokens_to_words_alignment = align.y2x.data
        
        # list of lists of length len words, 
        # each element contains the token indices that map to a word
        tokens_to_word_list = group_tokens(tokens_to_words_alignment)
        
        assert len(tokens_to_word_list) == len(words), print("Alignment failed")

        activity_word_level_embed = []  
        activity_word_level_pos = []
        activity_word_level_pos_embed = []
        activity_word_level = []
        
        for idx, w in enumerate(tokens_to_word_list):
            
            is_punc = is_punctuation(words[idx])
                    
            if len(w) > 1:
                # take the mean of tokens within a word if it has multiple tokens 
                word_activity_embed = torch.squeeze(torch.mean(static_embed[w], axis=0))
                word_activity_pos = torch.squeeze(torch.mean(static_pos[w], axis=0))
                word_activity_pos_embed = torch.squeeze(torch.mean(static_embed_pos[w], axis=0))
                word_activity = torch.squeeze(torch.mean(outputs[:, w], axis=1))
            else:
                word_activity_embed = torch.squeeze(static_embed[w])
                word_activity_pos = torch.squeeze(static_pos[w])
                word_activity_pos_embed = torch.squeeze(static_embed_pos[w])
                word_activity = torch.squeeze(outputs[:, w])
                
            # don't add punctuation to static embeddings for trained embeddings since that
            # worsens the performance of static embeddings 
            if is_punc:
                activity_word_level.append(word_activity.cpu().detach().numpy())
            else:
                activity_word_level.append(word_activity.cpu().detach().numpy())
                activity_word_level_pos_embed.append(word_activity_pos_embed.cpu().detach().numpy())
                activity_word_level_pos.append(word_activity_pos.cpu().detach().numpy())
                activity_word_level_embed.append(word_activity_embed.cpu().detach().numpy())
                
        activity_word_level_pos_embed = np.array(activity_word_level_pos_embed)
        activity_word_level_pos = np.array(activity_word_level_pos)
        activity_word_level_embed = np.array(activity_word_level_embed)
        activity_word_level = np.array(activity_word_level)
        
        activity_sent, activity_sent_sp, static_activity_pos_embed, static_activity_pos, static_activity_embed = \
        pool_representations(dataset, activity_word_level, activity_word_level_pos_embed, activity_word_level_pos, activity_word_level_embed)
        
        non_averaged_static = activity_word_level_pos_embed
        
    return static_activity_pos_embed, static_activity_pos, static_activity_embed, activity_sent, activity_sent_sp, non_averaged_static


static_embed_activity = []
static_pos_embed_activity = []
static_pos_embed_activity_non_avg = []
static_pos_activity = []
contextual_activity = []
contextual_activity_sp = []
previous_text =  '' 
current_label = data_labels[0]
total_words = 0 
num_words_or_tokens = []

print("GENERATING ACTIVATIONS")

for txt, dl in zip(experiment_txt, data_labels):
    
    # remove right spaces
    txt = txt.rstrip()
    
    if dl != current_label:
        # reset the previous context 
        previous_text = ''
        current_label = dl
    
    current_text = f' {txt}'

    if dataset == 'fedorenko':
        current_text = current_text.replace('.', '')
    
    static_pos_embed_rep, static_pos_rep, static_embed_rep, contextual_rep, contextual_rep_sp, static_non_avg = get_model_activity(previous_text, 
                current_text, embedding_matrix, positional_matrix, tokenizer, model_str=model_str, dataset=dataset, save_words=save_words)

    previous_text += current_text

    static_pos_embed_activity.append(static_pos_embed_rep)
    static_pos_activity.append(static_pos_rep)
    static_embed_activity.append(static_embed_rep)
    contextual_activity.append(contextual_rep)
    contextual_activity_sp.append(contextual_rep_sp)
    static_pos_embed_activity_non_avg.append(static_non_avg)
    num_words_or_tokens.append(static_non_avg.shape[0])

if dataset == 'pereira' or dataset == 'blank':
    
    contextual_activity_stacked = np.stack(contextual_activity)
    contextual_activity_stacked_sp = np.stack(contextual_activity_sp)
    static_pos_embed_activity_stacked = np.stack(static_pos_embed_activity)
    static_pos_embed_activity_non_avgs_stacked = np.vstack(static_pos_embed_activity_non_avg)

    static_pos_activity_stacked = np.stack(static_pos_activity)
    static_embed_activity_stacked = np.stack(static_embed_activity)
    
    contextual_dict = {}
    contextual_dict_sp = {}
    for ln in range(contextual_activity_stacked.shape[1]):
        contextual_dict[f'layer_{ln}'] = contextual_activity_stacked[:, ln]
        contextual_dict_sp[f'layer_{ln}'] = contextual_activity_stacked_sp[:, ln]
    
elif dataset == 'fedorenko':
    
    static_embed_activity_stacked = np.vstack(static_embed_activity)
    static_pos_activity_stacked = np.vstack(static_pos_activity)
    static_pos_embed_activity_stacked = np.vstack(static_pos_embed_activity)
    contextual_activity_stacked = np.vstack(contextual_activity)
    
    contextual_dict = {}
    for ln in range(contextual_activity_stacked.shape[1]):
        contextual_dict[f'layer_{ln}'] = contextual_activity_stacked[:, ln]
    
    contextual_dict_sp = None
    
if save_words == True:
    word_str = '-word'
else:
    word_str = ''

savePath = f'{savePath}data_processed/{dataset}/LLM_acts'

# create folder to save data if it doesn't already exist
if os.path.isdir(savePath):
    pass
else:
    os.makedirs(savePath)
    
np.save(f'{savePath}sent_length{word_str}', num_words_or_tokens)

# last token method
np.savez(f'{savePath}/X_{model_str}{word_str}{model_num}', **contextual_dict)


if contextual_dict_sp is not None:
    np.savez(f'{savePath}/X_{model_str}-sp{word_str}', **contextual_dict_sp)
    
if dataset == 'blank' or dataset == 'pereira':
    np.savez(f'{savePath}/X_{model_str}-sp-static-non-avg{word_str}', **{'layer1': static_pos_embed_activity_non_avgs_stacked})
    np.savez(f'{savePath}/X_{model_str}-sp-static{word_str}', **{'layer1': static_pos_embed_activity_stacked})
    np.savez(f'{savePath}/X_{model_str}-sp-static-pos{word_str}', **{'layer1': static_pos_activity_stacked})
    np.savez(f'{savePath}/X_{model_str}-sp-static-embed{word_str}', **{'layer1': static_embed_activity_stacked})