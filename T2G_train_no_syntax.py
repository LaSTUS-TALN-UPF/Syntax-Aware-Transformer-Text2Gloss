# -*- coding: utf-8 -*-

import tensorflow as tf
import sentencepiece as spm
import numpy as np
import pickle

from time import time
import os, sys
sys.path.insert(0, "./utils/")

from data_generator import *
tf.config.set_visible_devices([], 'GPU') # Uncomment to execute in CPU

#%%############################################################################
'''                        EXPERIMENTAL SETTINGS                            '''
###############################################################################
WORKING_DIR = 'T2G_no_syntax'

SRC_TRAIN_FILENAME = './data/spoken_train.txt'     # FILENAME CONTAINING SRC TEXT
TGT_TRAIN_FILENAME = './data/glosses_train.txt'     # FILENAME CONTAINING TGT TEXT

SRC_EVAL_FILENAME = './data/spoken_dev.txt'     # FILENAME CONTAINING SRC TEXT
TGT_EVAL_FILENAME = './data/glosses_dev.txt'     # FILENAME CONTAINING TGT TEXT

SRC_TOKEN = '<SPOK>'
TGT_TOKEN = '<GLOSS>'

SPACY_MODEL = None


BATCH_SIZE = 64 
N_EPOCHS = 500
LR = 1e-5

VOCAB_SIZE = 3000
SCALE_DOWN_FACTOR = 4
EMBEDDING_DIM = 512
N_TAG_TOKEN = len(TAG_MAPPER)
tf.print(WORKING_DIR)
#%%############################################################################
'''                         CREATING CONFIG DICT                            '''
###############################################################################
variables = ['WORKING_DIR', 'SRC_TRAIN_FILENAME', 'TGT_TRAIN_FILENAME', 'SRC_EVAL_FILENAME',
             'TGT_EVAL_FILENAME', 'SRC_TOKEN', 'TGT_TOKEN', 'SPACY_MODEL', 'BATCH_SIZE',
             'N_EPOCHS', 'LR', 'VOCAB_SIZE', 'SCALE_DOWN_FACTOR', 'EMBEDDING_DIM', 'N_TAG_TOKEN']
exp_settings = {i:eval(i) for i in variables}

#%%############################################################################
'''                      CREATING WORKING DIRECTORY                         '''
###############################################################################

subdirs = ['trained_models']
if WORKING_DIR is None:
    from datetime import datetime
    now = datetime.now() 
    WORKING_DIR = now.strftime("%m_%d_%H_%M")

list_dir = os.listdir()
if not WORKING_DIR in list_dir:
    os.mkdir(WORKING_DIR)

handled_dirs = {}
list_dir = os.listdir(WORKING_DIR)
for d in subdirs:
    if not d in list_dir:
        os.mkdir(WORKING_DIR+'/'+d)
    handled_dirs[d] = WORKING_DIR+'/'+d

with open(WORKING_DIR+'/exp_settings.dict', 'w') as f:
    f.write(str(exp_settings))

#%%############################################################################
'''                      ACCOMODATING TRAIN-EVAL DATA                       '''
###############################################################################

with open(SRC_TRAIN_FILENAME, 'rb') as f:
    src_train = f.read().decode().split('\n')
with open(TGT_TRAIN_FILENAME, 'rb') as f:
    tgt_train = f.read().decode().split('\n')

with open(SRC_EVAL_FILENAME, 'rb') as f:
    src_dev = f.read().decode().split('\n')
with open(TGT_EVAL_FILENAME, 'rb') as f:
    tgt_dev = f.read().decode().split('\n')


#%%############################################################################
'''                          TRAINING SENTENCE PIECE                        '''
###############################################################################
eos_token = '<end>'
bos_token = '<start>'
pad_token = '<pad>'
special_tokens = [bos_token, eos_token, pad_token,
                  TGT_TOKEN, SRC_TOKEN]

with open('./{}/training_text.txt'.format(WORKING_DIR), 'w', encoding = 'utf-8') as f:
    f.write('\n'.join(src_train+tgt_train))
    
spm.SentencePieceTrainer.train(input = './{}/training_text.txt'.format(WORKING_DIR),
                               model_prefix = './{}/test_{}'.format(WORKING_DIR, VOCAB_SIZE),
                                vocab_size=VOCAB_SIZE,
                                user_defined_symbols = ','.join(special_tokens))


#%%############################################################################
'''                          PREPARING DATA GENERATOR                       '''
###############################################################################

sp = spm.SentencePieceProcessor(model_file='./{}/test_{}.model'.format(WORKING_DIR, VOCAB_SIZE))
special_ids = [i[1] for i in sp.tokenize(special_tokens)]

train_generator = data_generatorV3(sp, src_train, tgt_data = tgt_train,
                 # TRAINING VARIABLES
                 batch_size=BATCH_SIZE, shuffle = True, training = True, 
                 
                 # SYNTAX TAGGING
                 include_syntax = False, spacy_model = SPACY_MODEL,  
                 
                 # CONTROL TOKEN VARIABLES
                 src_lan_id = special_ids[3], 
                 tgt_lan_id = special_ids[4],
                 pad_id = special_ids[2],
                 bos_id = special_ids[0],
                 eos_id = special_ids[1],)



eval_generator = data_generatorV3(sp, src_dev, tgt_data = tgt_dev,
                 # TRAINING VARIABLES
                 batch_size=BATCH_SIZE, shuffle = False, training = False, 
                 
                 # SYNTAX TAGGING
                 include_syntax = False, spacy_model = SPACY_MODEL,  
                 
                 # CONTROL TOKEN VARIABLES
                 src_lan_id = special_ids[3], 
                 tgt_lan_id = special_ids[4],
                 pad_id = special_ids[2],
                 bos_id = special_ids[0],
                 eos_id = special_ids[1],)


#%%############################################################################
'''                     INITIALIZING TRANSFORMER MODEL                      '''
###############################################################################

from transformers.models.mbart.modeling_tf_mbart import (TFMBartEncoder, MBartConfig, 
                                                         TFSharedEmbeddings, TFMBartDecoder)

with open('./data/config_architecture.dict', 'rb') as f:
            config_dict = pickle.load(f)

# UPDATING DICTIONARY VALUES
mbarConfig_dict = MBartConfig(config_dict)

mbarConfig_dict.vocab_size = VOCAB_SIZE
mbarConfig_dict.d_model = EMBEDDING_DIM
mbarConfig_dict.max_position_embeddings = EMBEDDING_DIM

mbarConfig_dict.encoder_attention_heads  //=  SCALE_DOWN_FACTOR
mbarConfig_dict.encoder_ffn_dim //= SCALE_DOWN_FACTOR
mbarConfig_dict.encoder_layers //= SCALE_DOWN_FACTOR

mbarConfig_dict.decoder_attention_heads //= SCALE_DOWN_FACTOR
mbarConfig_dict.decoder_ffn_dim //= SCALE_DOWN_FACTOR
mbarConfig_dict.decoder_layers //= SCALE_DOWN_FACTOR

word_embedding_table = TFSharedEmbeddings(mbarConfig_dict.vocab_size, mbarConfig_dict.d_model)

encoder = TFMBartEncoder(mbarConfig_dict)
decoder = TFMBartDecoder(mbarConfig_dict)


enc_inputs_embeds =  word_embedding_table(tf.expand_dims([1], -1))*32
enc_hidden_states = encoder(input_ids = None, 
        inputs_embeds = enc_inputs_embeds, attention_mask=tf.expand_dims([1], -1))['last_hidden_state']
dec_inputs_embeds = word_embedding_table(tf.expand_dims([1], -1))

dec_hidden_states = decoder(input_ids=None,
        inputs_embeds=dec_inputs_embeds,
        encoder_hidden_states=enc_hidden_states,
        encoder_attention_mask=tf.expand_dims([1], -1),
        past_key_values=None,
        use_cache=False,
        training=False)

#%%############################################################################
'''                           SEQUENTIAL FORWARD                            '''
###############################################################################
optimizer = tf.keras.optimizers.Adam(lr=LR)

def forward_pass(enc_word_ids, enc_input_att, dec_input_ids, training=False):
    
    enc_inputs_embeds = word_embedding_table(enc_word_ids)*32.0

    enc_hidden_states = encoder(input_ids = None, training = training,
            inputs_embeds = enc_inputs_embeds, attention_mask=enc_input_att)['last_hidden_state']
    
    dec_inputs_embeds = word_embedding_table(dec_input_ids)*32.0 

    dec_hidden_states = decoder(input_ids=None,
            inputs_embeds=dec_inputs_embeds,
            encoder_hidden_states=enc_hidden_states,
            encoder_attention_mask=enc_input_att,
            past_key_values=None,
            use_cache=False,
            training=training)
    
    logits = word_embedding_table(dec_hidden_states[0], mode = 'linear')
    return logits


#%%############################################################################
'''                      SEQUENTIAL TRAINING LOOP                           '''
###############################################################################
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,                          
                                                                reduction=tf.keras.losses.Reduction.NONE) 
def loss_function(labels, logits, enc_mask):
  loss_ = loss_object(labels[:,1:], logits[:, :-1, :])
  mask = tf.math.logical_or(tf.not_equal(labels, tf.constant(special_ids[2])),
                           tf.cast(enc_mask, tf.bool)) 
  loss_masked = loss_*tf.cast(mask[:,:1], tf.float32)
  return tf.reduce_mean(loss_masked)

#from losses import cross_entropy_loss_label_smoothing
#loss_function = cross_entropy_loss_label_smoothing

print('Fitting!')

for epoch in range(1,N_EPOCHS+1):
    
    tf.print('Epoch {}/{}'.format(epoch, N_EPOCHS))
    t0 = time.time()
    loss = 0.0
    variables = word_embedding_table.trainable_variables+\
                    encoder.trainable_variables+decoder.trainable_variables
    for i in range(len(train_generator)):
        input_batch = train_generator[i]
        with tf.GradientTape() as tape:
            logits = forward_pass(input_batch[0][0],
                                  input_batch[0][1],
                                  input_batch[1],
                                  training = True)
                                       
            
            
            loss += loss_function(input_batch[1], logits, input_batch[0][1])

        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))        
    tf.print('Elapsed Time: {}'.format(time.time()-t0))
    tf.print('Training Loss: {}'.format(loss/train_generator.n_sentences))
    if not epoch%5:
        eval_loss = 0.0

        for i in range(len(eval_generator)):
            input_batch = eval_generator[i]
            logits = forward_pass(input_batch[0][0],
                                      input_batch[0][1],
                                      input_batch[1],
                                      training = True)
            eval_loss += loss_function(input_batch[1], logits)
        
        tf.print('Eval Loss: {}'.format(eval_loss/eval_generator.n_sentences))

        #### STORING MODEL WEIGHTS
        encoder_weights = encoder.get_weights()
        with open(handled_dirs['trained_models']+'/encoder_{}.weights'.format(epoch),'wb') as f:
            pickle.dump(encoder_weights, f)
        del encoder_weights
        
        decoder_weights = decoder.get_weights()
        with open(handled_dirs['trained_models']+'/decoder_{}.weights'.format(epoch),'wb') as f:
            pickle.dump(decoder_weights, f)    
        del decoder_weights
        
        emb_weights = word_embedding_table.get_weights()
        with open(handled_dirs['trained_models']+'/word_embeddings_{}.weights'.format(epoch),'wb') as f:
            pickle.dump(emb_weights, f)      
        del emb_weights
    
    #### SHUFFLE GENERATOR
    train_generator.on_epoch_end()
