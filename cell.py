# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:28:08 2017

@author: chuito
"""

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.contrib.rnn.python.ops.rnn_cell import _linear


class GRNNSRCell(RNNCell):
    def __init__(self, num_units, activation=tf.tanh, state_is_tuple=True):
        self._num_units = num_units
        self._activation = activation
        self._state_is_tuple = state_is_tuple

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "grnnsr_cell"):
            c, h = state
            with tf.variable_scope("gates"):
                u_c, u_h, r_c, r_w = array_ops.split(
                    split_dim=1,
                    num_split=4,
                    value=tf.sigmoid(_linear([inputs, c], 4 * self._num_units, True, 1.0))
                )
            with tf.variable_scope("inputs"):
                j_c = tf.tanh(_linear([inputs, r_c * c], self._num_units, True, scope="input_c"))
                w = 2 * tf.tanh(_linear(c * r_w, self._num_units, True, scope="weights"))
                j_h = w * tf.tanh(_linear(inputs, self._num_units, True, scope="input_h"))
            new_c = u_c * c + (1 - u_c) * j_c
            new_h = u_h * h + (1 - u_h) * j_h
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

class GRNNSPCell(RNNCell):
    def __init__(self, num_units, activation=tf.tanh, state_is_tuple=True):
        self._num_units = num_units
        self._activation = activation
        self._state_is_tuple = state_is_tuple

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "grnnsp_cell"):
            c, h = state
            with tf.variable_scope("gates"):
                u_c, u_h, r_c, r_w = array_ops.split(
                    split_dim=1,
                    num_split=4,
                    value=tf.sigmoid(_linear([inputs, c], 4 * self._num_units, True, 1.0))
                )
            with tf.variable_scope("inputs"):
                j_c = tf.tanh(_linear([inputs, r_c * c], self._num_units, True, scope="input_c"))
                j_h = tf.tanh(_linear(inputs, self._num_units, True, scope="input_h"))
            new_c = u_c * c + (1 - u_c) * j_c
            new_h = u_h * h + (1 - u_h) * j_h
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state