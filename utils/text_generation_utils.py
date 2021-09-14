# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 12:35:05 2021

@author: SNT
"""
import tensorflow as tf
import numpy as np


    

#%%############################################################################
'''                        BeamHypotheses (from Huggingface)                '''
###############################################################################

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

#%%############################################################################
'''                      Useful Helpers   (from Huggingface)                '''
###############################################################################

def _reorder_cache(past, beam_idx):
    if len(past) == 1:
        return past

    past_key_values = past[1]

    reordered_past = ()
    for layer_past_key_values in past_key_values:
        reordered_past += (
            tuple(tf.gather(layer_past_key_value, beam_idx) for layer_past_key_value in layer_past_key_values[:2])
            + layer_past_key_values[2:],
        )
    return (past[0], reordered_past)


def set_tensor_by_indices_to_value(tensor, indices, value):
    # create value_tensor since tensor value assignment is not possible in TF
    value_tensor = tf.zeros_like(tensor) + value
    return tf.where(indices, value_tensor, tensor)

def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty):
    # create logit penalties for already seen input_ids
    token_penalties = np.ones(tf.shape(logits))
    prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
    for i, prev_input_id in enumerate(prev_input_ids):
        logit_penalized = logits[i].numpy()[prev_input_id]
        logit_penalties = np.zeros(logit_penalized.shape)
        # if previous logit score is < 0 then multiply repetition penalty else divide
        logit_penalties[logit_penalized < 0] = repetition_penalty
        logit_penalties[logit_penalized > 0] = 1 / repetition_penalty
        np.put(token_penalties[i], prev_input_id, logit_penalties)
    return tf.convert_to_tensor(token_penalties, dtype=tf.float32)


def _extend_tensors(tensor_list, batch_size, num_beams, input_seq_len):
    extended_list = []
    for t in tensor_list:
        t = tf.broadcast_to(tf.expand_dims(t, 1),
                            (batch_size, num_beams, input_seq_len))
        
        t = tf.reshape(t, (batch_size * num_beams, input_seq_len))
        extended_list.append(t)
    return extended_list



