import numpy as np
import sys
sys.path.append('/data/LLMs/LMMS/')
from transformers_encoder import TransformersEncoder
from transformers import RobertaModel, RobertaTokenizer
from vectorspace import SensesVSM
import spacy
en_nlp = spacy.load('en_core_web_trf')  # required for lemmatization and POS-tagging 
import torch
from wn_utils import WN_Utils
wn_utils = WN_Utils()  # WordNet auxilliary methods (just for describing results)
basePath = '/data/LLMs/LMMS/'

'''
Python file used to generate LMMS word and sense embeddings.
'''


mode = 'word'
dataset = 'pereira'

# NLM/LMMS paths and parameters
vecs_path = f'{basePath}data/vectors/lmms-sp-wsd.roberta-large.vectors.txt'
wsd_encoder_cfg = {
    'model_name_or_path': 'roberta-large',
    'min_seq_len': 0,
    'max_seq_len': 512,
    'layers': [-n for n in range(1, 24 + 1)],  # all layers, with reversed indices
    'layer_op': 'ws',
    'weights_path': f'{basePath}data/weights/lmms-sp-wsd.roberta-large.weights.txt',
    'subword_op': 'mean'
}


print('Loading NLM and sense embeddings ...')  # (takes a while)
wsd_encoder = TransformersEncoder(wsd_encoder_cfg)
senses_vsm = SensesVSM(vecs_path, normalize=True)
print('Done')

model_str = 'roberta-large'
tokenizer = RobertaTokenizer.from_pretrained(model_str)
model = RobertaModel.from_pretrained(model_str)
embedding_size = 1024

embedding_matrix = model.get_input_embeddings().weight.data
positional_matrix = model.embeddings.position_embeddings.weight.data


# words to merge with the next token(s)
merge_word = {'brand_new': 'brand-new', 'all_-': 'all-night', 'united_states': \
                'united_states', 'ear_piercings': 'earring', 'tear_gas': 'tear_gas'}

# how many tokens to skip after merging words 
num_skip = {'brand-new': 1, 'all-night': 2, 'united_states': 1, 'earring': 1, 'tear_gas': 1}

def modify_pos_word(word, pos, next_word, dataset, merge_word=merge_word, num_skip=num_skip):
    
    '''
    Modifies some words / part of speech tags if they are not in LMMS.
    '''
    
    if dataset == 'federonko':
        
        return word, pos, 0
    
    elif dataset == 'pereira':
        
        # words to modify 
        update_words = {'artisanal': 'artisan', 'lawnmower': 'mower', 'fulltime': 'full-time', 
                        'waterbed': 'water_bed', 'airbed': 'air_mattress', 'videogame': 'video_game', '1970': 'seventies', 
                        'micromanager': 'manager', 'showcase': 'show_off', 'scissor': 'scissors', 'vikings': 'viking', 
                        'wildland': 'land', 'landform': 'terrain', 'stabbing': 'stab', 'shorter': 'short', 
                        'pedalling': 'pedal', 'freezing': 'frigid', 'tong': 'tongs', 'feet': 'ft'}
        
        # words to modify pos tagging 
        pos_tag = {'mild': 'ADJ', 'spacewalk': 'VERB', 'underwater': 'ADJ', 'artisan': 'NOUN', 
                'surrounding': 'ADJ', 'brand-new': 'ADJ', 'all-night': 'ADJ', 'underneath': 'ADV', 'polar': 'ADJ', 
                'stab': 'VERB', 'tear_gas': 'NOUN', 'dairy': 'NOUN', 'seventh': 'ADJ', 'tailless': 'ADJ'}
        
        modified = False
        
        # for words that are combined with their next token
        if f'{word}_{next_word}' in merge_word.keys():
            print(f"Merging: {word}_{next_word}")
            word = merge_word[f'{word}_{next_word}']
            modified = True
            
        # words we updated
        if word in update_words.keys():
            word = update_words[word]
            modified = True
        
        # get updated pos tag based on the (potentially) modified representation
        if word in pos_tag.keys():
            pos = pos_tag[word]
            
        # for merged or updated words, check if we should skip the next token(s)
        if word in num_skip.keys() and modified:
            skip_next = num_skip[word]
        else:
            skip_next = 0

        return word, pos, skip_next
        
def split_words_and_combine(word_list):
    
    split_word_dict = {'wildland': 'wild land', 
                       'landform': 'land form'}
    word_list_new = []
    for w in word_list:
        if w in split_word_dict.keys():
            print(w)
            word_list_new.extend(split_word_dict[w].split())
        else:
            word_list_new.append(w)
    
    sentence = ' '.join(word_list_new)
    sentence += '.'
    return sentence

def merge_tokens(tokens, dataset, merge_words=merge_word, num_skip_dict=num_skip):
    
    if dataset == 'federonko':
        return tokens 
    
    elif dataset == 'pereira':
        new_tokens = []
        num_skip = 0
        for idx, t in enumerate(tokens):
            
            if num_skip > 0:
                num_skip -= 1
                continue
            try:
                next_token = tokens[idx+1]
            except:
                next_token = 'none'

            if f'{t.lower()}_{next_token.lower()}' in merge_words.keys():
                modified_word = merge_words[f'{t.lower()}_{next_token.lower()}']
                print("MODIFYING FOR ROBERTA: ", modified_word)
                num_skip = num_skip_dict[modified_word]
                contextual_token = ''
                for i in range(num_skip+1):
                    contextual_token += tokens[idx+i]
                new_tokens.append(contextual_token)
            else:
                new_tokens.append(t)

        return new_tokens
    
# slightly modified sentences from peirera to get sense embeddings 
# load pereria text
sentences_path = f"/home3/ebrahim/what-is-brainscore/{dataset}_data/sentences_ordered_dereferenced.txt"
with open(sentences_path, "r") as file:
    # Read the contents line by line into a list
    sentences_text = [line.strip() for line in file]

data_labels= np.load(f'/home3/ebrahim/what-is-brainscore/data_processed/{dataset}/data_labels_{dataset}.npy')
dataset_representations = []
filler_sentence_representations = []
embed_pos_reps = []
num_words_sentence = []
no_sense_words_all = [] 

passage_text = ''
passage_label = data_labels[0]

no_sense_idxs_dict = {}

sentences_modified = []

for j, (sentence, label) in enumerate(zip(sentences_text, data_labels)):
    
    if dataset == 'federonko':
        sentence = sentence.replace('.', '')

    # store lemma and part of speech tags for current sentence
    lemma_arr = []
    postag_arr = []
    
    # split sentence into words + punctuation 
    doc = en_nlp(str(sentence))
    tokens = [t.text for t in doc]
    
    merged_tokens = merge_tokens(tokens, dataset)
    
    # split entire passage into words + punctuation 
    #doc_passage = en_nlp(str(passage_text))
    #tokens_passage = [t.text for t in doc_passage]
    
    num_words_sentence.append(len(tokens))

    # skip the end tokens of words that are combined
    skip_words = []
     
    if j % 10 == 0:
        print(j)
        
    # retrieve contextual embeddings for the entire current passage, and then extract tokens 
    # corresponding to the current sentence 
    if mode == 'sense':
        ctx_embeddings = wsd_encoder.token_embeddings([merged_tokens])[0]

    skip_next = 0
    
    for i, d in enumerate(doc):
    
        if skip_next > 0:
            print("Skipping: ", d)
            skip_next = skip_next - 1
            continue

        # get next word to check if the word should be combined with the next word
        # based on the modify_join_tokens function 
        try:
            next_tok =  str(d.nbor())
        except:
            next_tok = 'NONE'
            
        pos = d.pos_
            
        # get updated, lemma, pos, and whether to skip any words
        word, pos, skip_next = modify_pos_word(d.lemma_.lower(), pos, next_tok.lower(), dataset)
        
        # update sentence information 
        lemma_arr.append(word)
        postag_arr.append(pos)
    
    if '\n' in lemma_arr:
        lemma_arr.remove('\n')
        postag_arr.remove('SPACE')

    if mode == 'sense':
        if len(lemma_arr) != len(ctx_embeddings):
            print("Words not aligned")
            breakpoint()
            
        sentence_embeds, no_sense_words, no_sense_idxs = senses_vsm.get_sense_embeddings_sentence(lemma_arr, postag_arr, ctx_embeddings)
        
    elif mode == 'POS':
        sentence_embeds = senses_vsm.get_pos_embeddings_sentence(lemma_arr, postag_arr)
        
    elif mode == 'word':
        sentence_embeds = senses_vsm.get_word_embeddings_sentence(lemma_arr)
        
    dataset_representations.append(np.sum(np.stack(sentence_embeds),axis=0))


savePath = f'/home3/ebrahim/what-is-brainscore/data_processed/{dataset}/'
if mode == 'sense':
    np.savez(f'{savePath}/X_sense', **{'layer1': np.stack(dataset_representations)})
elif mode =='POS':
    np.savez(f'{savePath}/X_POS', **{'layer1': np.stack(dataset_representations)})
elif mode == 'word':
    np.savez(f'{savePath}/X_WORD', **{'layer1': np.stack(dataset_representations)})
