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
WORKING_DIR = 'T2G_syntax'                       # NAME OF THE WORKING SPACE
tf.print(WORKING_DIR)

with open(WORKING_DIR+'/exp_settings.dict', 'r') as f:
    exp_settings = eval(f.read())

for i in exp_settings:
    if i == 'WORKING_DIR': assert exp_settings[i] == WORKING_DIR
    exec("{} = exp_settings['{}']".format(i,i))
    
TRAINED_MODEL_WEIGHTS = [str(i) for i in range(5,16, 5)] # EPOCHS FOR WHICH COMPUTE THE METRICS
BATCH_SIZE = 128                              # BATCH SIZE USED FOR GENERATION

GEN_SRC_FILENAME = 'data/spoken_test.txt'
GEN_TGT_FILENAME = 'data/glosses_test.txt'
#%%############################################################################
'''                      CREATING WORKING DIRECTORY                         '''
###############################################################################

subdirs = ['generated_text', 'trained_models']
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


#%%############################################################################
'''                         ACCOMODATING ES-EN DATA                         '''
###############################################################################

with open(GEN_SRC_FILENAME, 'rb') as f:
    src_data = f.read().decode().split('\n')
N_SAMPLES_FOR_EXPERIMENT = len(src_data)
with open(GEN_TGT_FILENAME, 'rb') as f:
    tgt_data = f.read().decode().split('\n')
    
    
#%%############################################################################
'''                          PREPARING DATA GENERATOR                       '''
###############################################################################
sp = spm.SentencePieceProcessor(model_file='{}/test_{}.model'.format(WORKING_DIR,
                                                                     VOCAB_SIZE))

eos_token = '<end>'
bos_token = '<start>'
pad_token = '<pad>'
special_tokens = [bos_token, eos_token, pad_token,
                  SRC_TOKEN, TGT_TOKEN]
special_ids = [i[1] for i in sp.tokenize(special_tokens)]

test_data_gen = data_generatorV3(sp, src_data, tgt_data = tgt_data,
                 # TRAINING VARIABLES
                 batch_size=BATCH_SIZE, shuffle = False, training = False, 
                 
                 # SYNTAX TAGGING
                 include_syntax = True, spacy_model = SPACY_MODEL,  
                 
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
dep_embedding_table = TFSharedEmbeddings(N_TAG_TOKEN, mbarConfig_dict.d_model)

encoder = TFMBartEncoder(mbarConfig_dict)
decoder = TFMBartDecoder(mbarConfig_dict)

word_emb = word_embedding_table(tf.expand_dims([1], -1))*32.0
dep_emb = dep_embedding_table(tf.expand_dims([1], -1))*32.0

enc_inputs_embeds = word_emb+dep_emb
enc_hidden_states = encoder(input_ids = None, 
        inputs_embeds = enc_inputs_embeds, attention_mask=tf.expand_dims([1], -1))['last_hidden_state']
dec_inputs_embeds = word_embedding_table(tf.expand_dims([1], -1))*32.0

dec_hidden_states = decoder(input_ids=None,
        inputs_embeds=dec_inputs_embeds,
        encoder_hidden_states=enc_hidden_states,
        encoder_attention_mask=tf.expand_dims([1], -1),
        past_key_values=None,
        use_cache=False,
        training=False)

#%%############################################################################
'''              GENERATE_SEQ FUNCTION (MODIFIED FROM HUGGINGFACE)          '''
###############################################################################
from text_generation_utils import BeamHypotheses, _reorder_cache, _create_next_token_logits_penalties, _extend_tensors, set_tensor_by_indices_to_value
def beam_search_generation( enc_word_ids, enc_dep_ids, enc_input_att,
                            target_lan_token_id = special_ids[4],
                            pad_token_id = special_ids[2],
                            bos_token_id = special_ids[0],
                            eos_token_id = special_ids[1],
                            
                            num_beams = 5,
                            early_stopping = True,
                            length_penalty = 1.0,
                            min_length= 0,
                            max_length = 200,
                            repetition_penalty = 1.0,
                            temperature = 1.0):
    
    ''' VARIABLE INITIALIZATION '''
    batch_size = enc_word_ids.shape[0]
    input_seq_len = int(tf.shape(enc_word_ids)[-1])
    vocab_size = word_embedding_table.vocab_size
    
    # TOKENS IDS INITIALIZATION
    target_lan_token_id = SPECIAL_TOKEN_2_IDS["es_XX"] if target_lan_token_id == None else target_lan_token_id
    eos_token_id = SPECIAL_TOKEN_2_IDS['</s>'] if eos_token_id == None else eos_token_id
    pad_token_id = SPECIAL_TOKEN_2_IDS['<pad>'] if pad_token_id == None else pad_token_id
    bos_token_id = SPECIAL_TOKEN_2_IDS['</s>']  if bos_token_id == None else bos_token_id 
    
    ''' ENCODING INPUT TOKENS '''
    encoder_embs = (word_embedding_table(enc_word_ids)+
                                        dep_embedding_table(enc_dep_ids))*32.

    encoder_outputs = encoder(None, inputs_embeds = encoder_embs,
                              attention_mask=enc_input_att)    
    
    [enc_word_ids, enc_dep_ids, enc_input_att] = _extend_tensors([enc_word_ids, enc_dep_ids, enc_input_att], batch_size, num_beams, input_seq_len)
    
    
    # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
    expanded_batch_idxs = tf.reshape(
        tf.repeat(tf.expand_dims(tf.range(batch_size), -1), repeats=num_beams , axis=1),
        shape=(-1,),
    )
        
    # expand encoder_outputs
    encoder_outputs = (tf.gather(encoder_outputs[0], expanded_batch_idxs, axis=0),)
    
    
    # Generating Beam Hypothesis
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]
    
    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    beam_scores = tf.concat([tf.zeros((batch_size, 1), dtype=tf.float32),
                             tf.ones((batch_size, num_beams - 1), dtype=tf.float32) * (-1e9)], -1)
    beam_scores = tf.reshape(beam_scores, (batch_size * num_beams,))
    # True when sentences are completely generated
    done = [False for _ in range(batch_size)]
    
    
    past = encoder_outputs    
    encoder_hidden_states=encoder_outputs[0]
    encoder_attention_mask = enc_input_att 
    input_ids = (tf.ones((batch_size * num_beams, 1),
                                 dtype=tf.int32)* bos_token_id) 
    past_key_values = None
    cur_len = 1
    
    
    
    while cur_len < max_length:
    
        decoder_input_ids = tf.expand_dims(tf.gather(input_ids,cur_len-1,axis = -1), -1)
        decoder_token_embs = word_embedding_table(decoder_input_ids)*32.0       
        outputs = decoder(None,
                inputs_embeds=decoder_token_embs,
                encoder_attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states, #Deshacer Tuple?
                return_dict = True,
                past_key_values = past_key_values
                )    
        
        logits = word_embedding_table(outputs[0], mode = 'linear')
        next_token_logits = logits[:, -1, :]  # (batch_size * num_beams, vocab_size)
        past = outputs[1]
        
        
        ''' SETTING EOS AND BOS TOKENS & APPLYING PENALTIES (ADJUSTING LOGITS) '''
        if repetition_penalty != 1.0:
            next_token_logits_penalties = _create_next_token_logits_penalties(
                input_ids, next_token_logits, repetition_penalty
            )
            next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)        
        
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature        
        
        # ADJUSTING LOGITS
        if cur_len == 1:
            vocab_range = tf.constant(range(vocab_size))
            next_token_logits = tf.where(vocab_range != target_lan_token_id, -1e8, next_token_logits)
        elif cur_len == max_length - 1:
            vocab_range = tf.constant(range(vocab_size))
            next_token_logits = tf.where(vocab_range != eos_token_id, -1e8, next_token_logits)
    
    
    
        #calculate log softmax score
        scores = tf.nn.log_softmax(next_token_logits, axis=-1)  # (batch_size * num_beams, vocab_size)
        # set eos token prob to zero if min_length is not reached
        if cur_len < min_length:
            # create eos_token_id boolean mask
            num_batch_hypotheses = batch_size * num_beams
    
            is_token_logit_eos_token = tf.convert_to_tensor(
                [True if token is eos_token_id else False for token in range(vocab_size)], dtype=tf.bool
            )
            eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token, [num_batch_hypotheses, vocab_size])
            scores = set_tensor_by_indices_to_value(scores, eos_token_indices_mask, -float("inf"))
    
        # if do_sample == False
        # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
        next_scores = scores + tf.broadcast_to(
            beam_scores[:, None], (batch_size * num_beams, vocab_size)
        )  # (batch_size * num_beams, vocab_size)
    
        # re-organize to group the beam together (we are keeping top hypothesis across beams)
        next_scores = tf.reshape(
            next_scores, (batch_size, num_beams * vocab_size)
        )  # (batch_size, num_beams * vocab_size)
    
        next_scores, next_tokens = tf.math.top_k(next_scores, k=2 * num_beams, sorted=True)
    
        # next batch beam content
        next_batch_beam = []
    
        # for each sentence create the possible Hyposthesis
        for batch_idx in range(batch_size):
    
            # if we are done with this sentence
            if done[batch_idx]:
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue
    
            # next sentence beam content
            next_sent_beam = []
    
            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])):
                
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size
    
                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence or last iteration
                if (eos_token_id is not None) and (token_id.numpy() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        tf.identity(input_ids[effective_beam_id]), beam_token_score.numpy()
                    )
                else:
                    # add next predicted token if it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
    
                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break
            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                tf.reduce_max(next_scores[batch_idx]).numpy(), cur_len
            )
            # update next beam content
            next_batch_beam.extend(next_sent_beam)
    
        # stop when we are done with each sentence
        if all(done):
            break
    
        beam_scores = tf.convert_to_tensor([x[0] for x in next_batch_beam], dtype=tf.float32)
        beam_tokens = tf.convert_to_tensor([x[1] for x in next_batch_beam], dtype=tf.int32)
        beam_idx = tf.convert_to_tensor([x[2] for x in next_batch_beam], dtype=tf.int32)
    
        # re-order batch and update current length
        input_ids = tf.stack([tf.identity(input_ids[x, :]) for x in beam_idx])
        input_ids = tf.concat([input_ids, tf.expand_dims(beam_tokens, 1)], axis=-1)
        # re-order internal states & Initialize next forward values
        encoder_outputs, past_key_values = _reorder_cache(past, beam_idx)        
        cur_len += 1
    
    
    
    ''' FINILAZING GENERATION '''
    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(batch_size):
        # Add all open beam hypothesis to generated_hyps
        if done[batch_idx]:
            continue
        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).numpy().item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert tf.reduce_all(
                next_scores[batch_idx, :num_beams] == tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]
            )
    
        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].numpy().item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    
    # select the best hypotheses
    sent_lengths_list = []
    best = []
    
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        best_hyp = sorted_hyps.pop()[1]
        sent_lengths_list.append(len(best_hyp))
        best.append(best_hyp)
    
    
    sent_lengths = tf.convert_to_tensor(sent_lengths_list, dtype=tf.int32)
    
    # shorter batches are filled with pad_token
    if tf.reduce_min(sent_lengths).numpy() != tf.reduce_max(sent_lengths).numpy():
        sent_max_len = min(tf.reduce_max(sent_lengths).numpy() + 1, max_length)
        decoded_list = []
    
        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            # if sent_length is max_len do not pad
            if sent_lengths[i] == sent_max_len:
                decoded_slice = hypo
            else:
                # else pad to sent_max_len
                num_pad_tokens = sent_max_len - sent_lengths[i]
                padding = pad_token_id * tf.ones((num_pad_tokens,), dtype=tf.int32)
                decoded_slice = tf.concat([hypo, padding], axis=-1)
    
                # finish sentence with EOS token
                if sent_lengths[i] < max_length:
                    decoded_slice = tf.where(
                        tf.range(sent_max_len, dtype=tf.int32) == sent_lengths[i],
                        eos_token_id * tf.ones((sent_max_len,), dtype=tf.int32),
                        decoded_slice,
                    )
            # add to list
            decoded_list.append(decoded_slice)
    
        decoded = tf.stack(decoded_list)
    else:
        # none of the hypotheses have an eos_token
        decoded = tf.stack(best)
    return decoded

#%%############################################################################
'''                            GENERATION PROCESS                           '''
###############################################################################

for m in TRAINED_MODEL_WEIGHTS:
    # Finetuned 
    with open(handled_dirs['trained_models']+'/encoder_{}.weights'.format(m),'rb') as f:
        weights = pickle.load(f)
    encoder.set_weights(weights)
    
    with open(handled_dirs['trained_models']+'/decoder_{}.weights'.format(m),'rb') as f:
        weights = pickle.load(f)
    decoder.set_weights(weights)
    
    with open(handled_dirs['trained_models']+'/word_embeddings_{}.weights'.format(m),'rb') as f:
        weights = pickle.load(f)
    word_embedding_table.set_weights(weights)         
    
    with open(handled_dirs['trained_models']+'/dep_embeddings_{}.weights'.format(m),'rb') as f:
        weights = pickle.load(f)
    dep_embedding_table.set_weights(weights) 
    
    generated_text = ['  '*20 for _ in range(test_data_gen.n_sentences)]
    sentence_counter = 0
    t0 = time.time()
    for i in range(len(test_data_gen)):
        tf.print('Generation Batch {}/{}'.format(i+1, len(test_data_gen)))
        
        (enc_word_ids,enc_input_att, enc_dep_ids), dec_input_ids  = test_data_gen[i]
        generated_tokens = beam_search_generation(enc_word_ids, enc_dep_ids, enc_input_att)
        n_gen_sentences = generated_tokens.shape[0]
        
        generated_text[sentence_counter:sentence_counter+n_gen_sentences] = test_data_gen.detokenize(generated_tokens, clean=True)
        sentence_counter += n_gen_sentences
    gen_time = time.time()-t0
    tf.print('Elapsed Time: {}'.format(gen_time))
    with open(handled_dirs['generated_text']+'/{}.txt'.format(m), 'w', encoding = 'utf-8') as f:
        f.write('\n'.join(generated_text))
