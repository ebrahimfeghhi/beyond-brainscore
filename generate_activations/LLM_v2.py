import spacy
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
import torch
import re
from spacy.tokenizer import Tokenizer
from spacy.training import Alignment
device_number = 0
device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

import os
from interp_funcs import separate_words_commas_periods, group_tokens, is_punctuation
import argparse

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModel
os.environ["TRUST_REMOTE_CODE"] = "True"

'''
This script is used to generate activations from LLMs (GPT2-XL and RoBERTa)
for the Pereira, Fedorenko, and Blank datasets. 
'''

parser = argparse.ArgumentParser(description="Your program description")
parser.add_argument("--untrained", action='store_true', default=False, help="If true, generate activations for untrained model.")
parser.add_argument("--model", type=str, help="Model to generate activations for")
parser.add_argument("--model_num", type=str, default='', help="Seed number if specified")
parser.add_argument("--dataset", type=str, default='pereira', help="pereira, fedorenko, or blank")
parser.add_argument("--decontext", action="store_true", default=False, help='If true, run each sentence in isolation for Pereira')

args = parser.parse_args()
basePath = ''
savePath = '/data/LLMs/'
dataset = args.dataset
model_str = args.model
untrained = args.untrained
model_num = args.model_num
decontext = args.decontext
print("Untrained: ", untrained)
print("generating activations for: ", model_str)

basePath_data = '/data/LLMs/data_processed/'


def generate_activations(basePath, savePath, dataset, model_str, untrained, model_num, decontext):

    masked_lm_list = [
    "distilbert/distilbert-base-uncased",
    "google-bert/bert-base-uncased",
    "google-bert/bert-base-multilingual-cased",
    "google-bert/bert-large-uncased",
    "google-bert/bert-large-uncased-whole-word-masking",
    "distilbert/distilroberta-base",
    "FacebookAI/roberta-base",
    "FacebookAI/roberta-large",
    "FacebookAI/xlm-mlm-enfr-1024",
    "FacebookAI/xlm-clm-enfr-1024",
    "FacebookAI/xlm-mlm-xnli15-1024",
    "FacebookAI/xlm-mlm-100-1280",
    "FacebookAI/xlm-mlm-en-2048",
    "FacebookAI/xlm-roberta-base",
    "FacebookAI/xlm-roberta-large",
    "albert/albert-base-v1",
    "albert/albert-base-v2",
    "albert/albert-large-v1",
    "albert/albert-large-v2",
    "albert/albert-xlarge-v1",
    "albert/albert-xlarge-v2",
    "albert/albert-xxlarge-v1",
    "albert/albert-xxlarge-v2",
    ]

    causal_lm_list = [
    "transfo-xl/transfo-xl-wt103",
    "xlnet/xlnet-base-cased",
    "xlnet/xlnet-large-cased",
    "openai-community/openai-gpt",
    "distilbert/distilgpt2",
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "openai-community/gpt2-large",
    "openai-community/gpt2-xl",
    ]

    other_models_list = [
    "Salesforce/ctrl"
    ]

    s2s_lm_list = [
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google-t5/t5-3b",
    ]

    rwkv_lm_list = [
    "RWKV/rwkv-4-169m-pile",
    "RWKV/rwkv-4-430m-pile",
    "RWKV/rwkv-4-1b5-pile",
    "RWKV/rwkv-4-3b-pile"
    ]

    mamba_lm_list = [
    "state-spaces/mamba-130m-hf",
    "state-spaces/mamba-370m-hf",
    "state-spaces/mamba-790m-hf",
    "state-spaces/mamba-1.4b-hf",
    "state-spaces/mamba-2.8b-hf"
    ]

    llama_lm_list = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct"
    ]

    causal_lm_list = causal_lm_list + rwkv_lm_list + mamba_lm_list + llama_lm_list


    def count_words(text):
        # Split the text into words based on spaces and count them
        words = text.split()
        return len(words)

    # load linguistic stimuli 
    if dataset == 'pereira':
        pereira_path = f"{basePath_data}{dataset}/text/sentences_ordered.txt"
        with open(pereira_path, "r") as file:
            # Read the contents line by line into a list
            experiment_txt = [line.strip() for line in file]
        # each element is exp-passage_name-passage_num-first half or second half
        data_labels = np.load(f"{basePath_data}{dataset}/dataset/data_labels_{dataset}.npy") 
        
    if dataset == 'fedorenko':
        
        fed_path = f"{basePath_data}{dataset}/text/sentences_ordered.txt"
        
        with open(fed_path, "r") as file:
            # Read the contents line by line into a list
            experiment_txt = [line.strip() for line in file]

        words_list = []
        for sentence in experiment_txt:
            for word in sentence.split():
                words_list.append(word)
        experiment_txt = words_list
        
        data_labels = np.repeat(np.arange(52), 8)

    if dataset == 'blank':
        blank_data = np.load(f"{basePath_data}{dataset}/text/story_data_dict.npz")
        experiment_txt = []
        data_labels = []
        for key, val in blank_data.items():
            experiment_txt.extend(val)
            data_labels.extend(np.repeat(key, len(val)))
        
    # load models 
    if '/' not in model_str:
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

    if model_str in causal_lm_list:
        model = AutoModelForCausalLM.from_pretrained(model_str)
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model.eval()
        model = model.to(device)
        embedding_matrix = None
        positional_matrix = None

    if model_str in masked_lm_list:
        model = AutoModelForMaskedLM.from_pretrained(model_str)
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model.eval()
        model = model.to(device)
        embedding_matrix = None
        positional_matrix = None

    if model_str in s2s_lm_list:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_str)
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model.eval()
        model = model.encoder
        model = model.to(device)
        embedding_matrix = None
        positional_matrix = None
        
    def pool_representations(contextual_embeddings):
        
        '''
        :param ndarray contextual embeddings: contextual embeddings from an LLM,
        of shape num_tokens/num_words x num_layers x embed_size 
        '''

        # last token pooling
        activity_sent = contextual_embeddings[-1]
        activity_sent_sp = np.sum(contextual_embeddings, axis=0)
        activity_sent_mp = np.mean(contextual_embeddings, axis=0)
        
        return activity_sent, activity_sent_sp, activity_sent_mp
            
        
    def get_model_activity(previous_text, current_text, embedding_matrix, 
                                positional_matrix, tokenizer, model_str, dataset, 
                                max_context_size=512):
        
        '''
        :param str previous_text: linguistic stimuli which serves as previous context
        :param str current_text: the linguistic stimuli to obtain activations for
        :param torch tensor embedding_matrix: static embedding matrix
        :param torch tensor positional_matrix: static positional matrix
        :param tokenizer: tokenize sentence
        :param str model_str: model used to generate activations
        :param str dataset: neural dataset 
        :param int max_context_size: max context size for LLM (in tokens)
        '''

        # separately tokenize current and previous tokens so that it's easier
        # to sum/avg pool across current tokens 
        curr_tokens = tokenizer.tokenize(current_text)
        num_ct = len(curr_tokens)
        prev_tokens = tokenizer.tokenize(previous_text)
        full_text = previous_text + current_text
        tokens = prev_tokens + curr_tokens
        tokens_all = tokenizer.tokenize(full_text)
        
        # make sure separating current and previous text
        # doesn't lead to different tokenization outputs
        if tokens != tokens_all:
            if 'xlm' not in model_str:
                print("Be caferful, tokenization is different")
                breakpoint()
        effective_context_size = max_context_size - len(tokenizer('')['input_ids'])
        tokens = tokens[-effective_context_size:]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # append special tokens 
        token_ids = tokenizer.build_inputs_with_special_tokens(token_ids)
        
        tensor_input = torch.tensor([token_ids])
        tensor_input = tensor_input.to(device)    
        #global counter, counter2
        #if 'counter' not in globals():
        #    counter = 0
        #    counter2 = 0
        with torch.no_grad():
            #print(counter, counter2, tensor_input.shape)
            #counter += 1
            #counter2 += tensor_input.shape[1]
            
            if model_str in ['FacebookAI/xlm-mlm-enfr-1024','FacebookAI/xlm-clm-enfr-1024', 'FacebookAI/xlm-mlm-xnli15-1024']:
                language_id = tokenizer.lang2id["en"]
                langs = torch.tensor([language_id] * tensor_input.shape[1]).to(device)  # torch.tensor([0, 0, 0, ..., 0])
                #print(langs)

                # We reshape it to be of size (batch_size, sequence_length)
                langs = langs.view(1, -1)  # is now of shape [1, sequence_length] (we have a batch size of 1)
                outputs = model(tensor_input, langs=langs, output_hidden_states=True)
            else:
                outputs = model(tensor_input, output_hidden_states=True, output_attentions=True)
            outputs = outputs.hidden_states
            # number of layers x context size x embedding size
            outputs = torch.stack(outputs).squeeze()
            
        # remove special tokens because we only want to sum across 
        # words/punctuation marks for consistency across models
        nonspecial_idxs = [x not in tokenizer.all_special_ids for x in token_ids]
        #breakpoint()
        try:
            if len(token_ids) > 1:
                activity_sent_lt = outputs[:,-1,:].cpu().detach().numpy()
                outputs = outputs[:,nonspecial_idxs,:]
            else:
                activity_sent_lt = outputs[:,:].cpu().detach().numpy()
                

        except:
            breakpoint()

        # only take tokens corresponding to the current text
        if len(token_ids) > 1:
            outputs = outputs[:, -num_ct:]
        else:
            outputs = torch.unsqueeze(outputs, dim=1)
            
        activity_sent, activity_sent_sp, activity_sent_mp = \
        pool_representations(outputs.cpu().detach().numpy().swapaxes(0,1))

        #breakpoint()
        return activity_sent, activity_sent_sp, activity_sent_mp, activity_sent_lt



    contextual_activity = []
    contextual_activity_sp = []
    contextual_activity_mp = []
    contextual_activity_lt = []
    previous_text =  '' 
    current_label = data_labels[0]
    total_words = 0 
    num_words_or_tokens = []

    print("GENERATING ACTIVATIONS")

    for txt, dl in zip(experiment_txt, data_labels):

        # remove right spaces
        txt = txt.rstrip()
        
        # if new passage/sentence/story, then reset context
        # also reset context if running in decontextualized mode
        if dl != current_label or decontext:
            # reset the previous context 
            previous_text = ''
            current_label = dl

        if len(previous_text) == 0:
            current_text = txt
        else:
            current_text = f' {txt}'
        #if dataset == 'fedorenko':
        #    current_text = current_text.replace('.', '')
        
        contextual_rep, contextual_rep_sp, contextual_rep_mp, contextual_rep_lt = get_model_activity(previous_text, 
                    current_text, embedding_matrix, positional_matrix, tokenizer, model_str=model_str, dataset=dataset)

        previous_text += current_text


        contextual_activity.append(contextual_rep)
        contextual_activity_sp.append(contextual_rep_sp)
        contextual_activity_mp.append(contextual_rep_mp)
        contextual_activity_lt.append(contextual_rep_lt)

    contextual_activity_stacked = np.stack(contextual_activity)
    contextual_activity_stacked_sp = np.stack(contextual_activity_sp)
    contextual_activity_stacked_mp = np.stack(contextual_activity_mp)
    contextual_activity_stacked_lt = np.stack(contextual_activity_lt)

    contextual_dict = {}
    contextual_dict_sp = {}
    contextual_dict_mp = {}
    contextual_dict_lt = {}
    
    for ln in range(contextual_activity_stacked.shape[1]):
        contextual_dict[f'layer_{ln}'] = contextual_activity_stacked[:, ln]
        contextual_dict_sp[f'layer_{ln}'] = contextual_activity_stacked_sp[:, ln]
        contextual_dict_mp[f'layer_{ln}'] = contextual_activity_stacked_mp[:, ln]
        contextual_dict_lt[f'layer_{ln}'] = contextual_activity_stacked_lt[:, ln]

    if decontext:
        model_str = f"{model_str}-decontext"

    savePath = f'{savePath}data_processed/{dataset}/acts'

    # create folder to save data if it doesn't already exist
    if os.path.isdir(savePath):
        pass
    else:
        os.makedirs(savePath)
        
    np.save(f'{savePath}sent_length', num_words_or_tokens)

    # get the model name for naming the saved activations

    if '/' in model_str:
        model_str = model_str.split('/')[1]

    np.savez(f'{savePath}/X_{model_str}{model_num}', **contextual_dict)
    np.savez(f'{savePath}/X_{model_str}{model_num}-sp', **contextual_dict_sp)
    np.savez(f'{savePath}/X_{model_str}{model_num}-mp', **contextual_dict_mp)
    #np.savez(f'{savePath}/X_{model_str}{model_num}-lt', **contextual_dict_lt)

generate_activations(basePath_data, savePath, dataset, model_str, untrained, model_num, decontext)