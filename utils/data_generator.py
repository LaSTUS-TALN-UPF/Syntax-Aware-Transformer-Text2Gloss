# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 15:56:18 2021
@author: SNT

This script contains utils used in the paper

"""
import tensorflow as tf
import numpy as np

#%%############################################################################
'''                     USEFUL MAPPINGS FOR TOKENIZATION                    '''
###############################################################################

'''                 DICTIONARY TO MAP SYNTAX INFO TO TOKENS                 '''
SYNTAX_MAPPER = {
    'CC' :      0, #coordinating conjunction
    'CD' :      1, #cardinal digit
    'DT' :      2, #determiner
    'EX' :      3, #existential there (like: “there is” … think of it like “there exists”)
    'FW' :      4, #foreign word
    'IN' :      5, #preposition/subordinating conjunction
    'JJ' :      6, #adjective ‘big’
    'JJR':      7, #adjective, comparative ‘bigger’
    'JJS':      8, #adjective, superlative ‘biggest’
    'LS' :      9, #list marker 1)
    'MD' :      10, #modal could, will
    'NN' :      11, #noun, singular ‘desk’
    'NNS':      12, #noun plural ‘desks’
    'NNP':      13, #proper noun, singular ‘Harrison’
    'NNPS':     14, #proper noun, plural ‘Americans’
    'PDT' :     15, #predeterminer ‘all the kids’
    'POS' :     16, #possessive ending parent’s
    'PRP' :     17, #personal pronoun I, he, she
    'PRP$':     18, #possessive pronoun my, his, hers
    'RB'  :     19, #adverb very, silently,
    'RBR' :     20, #adverb, comparative better
    'RBS' :     21, #adverb, superlative best
    'RP'  :     22, #particle give up
    'TO'  :     23, # to go ‘to’ the store.
    'UH'  :     24, #interjection, errrrrrrrm
    'VB'  :     25, #verb, base form take
    'VBD' :     26, #verb, past tense took
    'VBG' :     27, #verb, gerund/present participle taking
    'VBN' :     28, #verb, past participle taken
    'VBP' :     29, #verb, sing. present, non-3d take
    'VBZ' :     30, #verb, 3rd person sing. present takes
    'WDT' :     31, #wh-determiner which
    'WP'  :     32, #wh-pronoun who, what
    'WP$' :     33, # possessive wh-pronoun whose
    'WRB' :     34, # wh-abverb where, when
    'PUNCT':    35, # Punctuations
    'OTHER':    36, # Other syntax element
    'PAD'  :    37, # Padding for syntax information
    'SYM'  :    38,
    }




import time, re, string

#%%############################################################################
'''                        DATA GENERATOR WITH OTHER                        '''
###############################################################################

class data_generatorV3(tf.keras.utils.Sequence):

    def __init__(self,  
                 
                 # DATA VARIABLES
                 tokenizer, src_data, tgt_data = None,
                 
                 # TRAINING VARIABLES
                 batch_size=64, shuffle = True, training = True,
                 max_seq_len = None,
                 
                 # CONTROL TOKEN VARIABLES
                 src_lan_id = 250004, 
                 tgt_lan_id = 250005,
                 pad_id = 1,
                 bos_id = 3,
                 eos_id = 2,
                 
                 # VARIABLES FOR AVANCED TAGGING
                 include_syntax = False,    
                 spacy_model = "en_core_web_sm",    # "de_core_news_sm"
                 ):
        assert len(src_data) == len(tgt_data), 'Source and target must have same length'
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.training = training
        self.include_syntax = include_syntax
        self.shuffle = shuffle
        
        #Setting special token values
        self.src_lan_token = src_lan_id
        self.tgt_lan_token = tgt_lan_id
        self.bos_token = bos_id
        self.eos_token = eos_id
        self.pad_token = pad_id
        self.control_tokens_list = [src_lan_id, tgt_lan_id, bos_id,
                                    eos_id, pad_id, 0]
        
        # PRECLEANING CORPUS - Deleting empty sentences
        # self.src_data = [sen for sen in self.src_data if sen != '' and sen != ' ']
        # if self.tgt_data != None:
        #     self.tgt_data = [sen for sen in self.tgt_data if sen != '' and sen != ' ']
        self.n_sentences = len(self.src_data)
        self.len_ = int(np.ceil(self.n_sentences/batch_size))
        self.indexes = np.arange(self.n_sentences)
        self.on_epoch_end()
        
        # PREPROCESSING SOURCE SENTENCES
        if self.include_syntax:
            #src_pos_tags = ['    '*self.seq_length for i in range(self.n_sentences)]
            
            src_dep_tags = ['    '*50 for i in range(self.n_sentences)]
            src_tokens = [[1]*50 for i in range(self.n_sentences)]
            
            sentence_counter = 0
            import spacy
            nlp = spacy.load(spacy_model)
            to_disable = [com for com in nlp.component_names if com not in ['tok2vec', 'parser']]            
            for sen in self.src_data:
                sen = self._prepare_sentence(sen)
                with nlp.select_pipes(disable=to_disable):
                    data = nlp(sen)
                words = [t.text for t in data]

                
                tokens = [self.tokenizer.tokenize(w) for w in words]
                token_length = [len(t) for t in tokens]
                src_tokens[sentence_counter] = [t for t_list in tokens for t in t_list]

                dep = [[TAG_MAPPER[t.dep_]]*l for t, l in zip(data, token_length)]
                src_dep_tags[sentence_counter] = [t for t_list in dep for t in t_list] 
                assert len(src_tokens[sentence_counter]) == len(src_dep_tags[sentence_counter])
                sentence_counter += 1     
                
            self.src_dep_tags = src_dep_tags
            self.src_tokens = src_tokens
        else:
            src_tokens = [[1]*50 for i in range(self.n_sentences)]
            sentence_counter = 0
            for sen in self.src_data:
                sen = self._prepare_sentence(sen)
                words = [w for w in sen.split(' ') if w != '']        
                tokens = [self.tokenizer.tokenize(w) for w in words]
                src_tokens[sentence_counter] = [t for t_list in tokens for t in t_list]
                sentence_counter += 1            
            self.src_tokens = src_tokens
            self.src_dep_tags = None
        
        # PREPROCESSING TARGET SENTENCES
        if tgt_data != None:
            tgt_tokens = [[1]*50 for i in range(self.n_sentences)]
            sentence_counter = 0
            for sen in self.tgt_data:
                sen = self._prepare_sentence(sen)
                words = [w for w in sen.split(' ') if w != '']        
                tokens = [self.tokenizer.tokenize(w) for w in words]
                tgt_tokens[sentence_counter] = [t for t_list in tokens for t in t_list]
                sentence_counter += 1            
            self.tgt_tokens = tgt_tokens
            
        
        
    def _prepare_sentence(self, sentence):
        # This function prepare the text for processing
        sentence = re.sub('([.,!?()\]\[;])', r' \1 ', sentence)
        sentence = re.sub('\s{2,}', ' ', sentence)  
        patterns1 = re.findall(r'(?<=\|).+?(?=\|)',
                              re.sub('([0-9][\s*][.,][\s*][0-9])', r'|\1|', sentence))
        
        patterns2 = re.findall(r'(?<=\|).+?(?=\|)',
                   re.sub('([a-z][\s*][.][\s*][a-z][\s*][.])', r'|\1|', sentence))
        
        patterns = [patterns1[i] for i in range(0,len(patterns1),2)] + \
                    [patterns2[i] for i in range(0,len(patterns2),2)]
        
        for pat in patterns:
            sentence = sentence.replace(pat, pat.replace(" ", ""))
    
        return sentence        

        
    def __len__(self):
        return self.len_ 
    
        
    def __getitem__(self, index):
        if index > self.len_-1:
            raise IndexError()
        
        if index == self.len_-1:
            indexes = self.indexes[index*self.batch_size:]
        else:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            
        src_ids = [self.src_tokens[i] for i in list(indexes)]
        
        header = [self.bos_token, self.src_lan_token]
        src_ids = [header+ids+[self.eos_token] for ids in src_ids]        
        
        max_length_src = max([len(ids) for ids in src_ids])
        max_length_src = max_length_src if self.max_seq_len == None else min(self.max_seq_len,max_length_src)
        
        att_mask = [[1]*len(ids)+[0]*(max_length_src-(len(ids))) for ids in src_ids]
        src_ids = [ids+[self.pad_token]*(max_length_src-len(ids)) for ids in src_ids] 
        src_dep_tags = None

        correct_none_lamb = lambda  x : x if x != None else 0
        src_ids = [list(map(correct_none_lamb,i)) for i in src_ids]    

        if self.include_syntax:
            src_dep_tags = [self.src_dep_tags[i] for i in list(indexes)]
            src_dep_tags = [[SYNTAX_MAPPER['PAD'], SYNTAX_MAPPER['PAD']]+tags+[SYNTAX_MAPPER['PAD']]
                            for tags in src_dep_tags]
            src_dep_tags = [tags+[SYNTAX_MAPPER['PAD']]*(max_length_src-(len(tags))) for tags in src_dep_tags]
            

        if self.training or self.tgt_data != None:
            tgt_ids = [self.tgt_tokens[i] for i in list(indexes)]
            header = [self.bos_token, self.tgt_lan_token]
            tgt_ids = [header+ids+[self.eos_token] for ids in tgt_ids]              
            
            max_length_tgt = max([len(ids) for ids in tgt_ids]+[max_length_src]) 
            max_length_tgt = max_length_tgt if self.max_seq_len == None else min(self.max_seq_len,max_length_tgt)
            
            
            tgt_ids = [ids+[self.pad_token]*(max_length_tgt-len(ids))
                       for ids in tgt_ids]
            tgt_ids = [list(map(correct_none_lamb,i)) for i in tgt_ids]  
              
            src_ids = src_ids if max_length_src > max_length_tgt \
                        else [ids+[self.pad_token]*(max_length_tgt-len(ids)) for ids in src_ids]      

            att_mask = att_mask if max_length_src > max_length_tgt \
                        else [a+[0]*(max_length_tgt-max_length_src) for a in att_mask]      
            
            outputs = [tf.convert_to_tensor(src_ids, dtype = tf.int32),
                        tf.convert_to_tensor(att_mask, dtype = tf.int32)]
            if self.include_syntax:
                
                src_dep_tags = src_dep_tags if max_length_src > max_length_tgt \
                        else [d+[SYNTAX_MAPPER['PAD']]*(max_length_tgt-max_length_src) for d in src_dep_tags]    
                outputs = outputs + [tf.convert_to_tensor(src_dep_tags, dtype = tf.int32)]
            return outputs, tf.convert_to_tensor(tgt_ids)
       
        else:
            outputs = [tf.convert_to_tensor(src_ids, dtype = tf.int32),
                       tf.convert_to_tensor(att_mask, dtype = tf.int32)]
            if self.include_syntax:
                outputs = outputs + [tf.convert_to_tensor(src_dep_tags, dtype = tf.int32)]
            return outputs
            
    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def detokenize(self, ids, clean = True):
        if clean:
            ids = [[int(i) for i in t if i not in self.control_tokens_list] for t in ids]
        else: 
            ids = [[int(i) for i in t if i != self.pad_token] for t in ids]
        return self.tokenizer.detokenize(ids)
        

#%%############################################################################
'''                              SPACY TAG GLOSSARY                         '''
###############################################################################

with open('./data/tag_mapper.dict', 'r') as f:
    TAG_MAPPER = eval(f.read())




