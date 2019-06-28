# -*- coding: utf-8 -*-
"""
Created on Feb 21  10:15  2017

@author: chuito
"""

import os
import tensorflow as tf

class BaseModel(object):
    """Abstract object representing an Reader model."""
    def __init__(self):
        self.name = "BaseModel"

    def get_model_dir(self):
        model_dir = self.name + "/" + self.dataset
        for attr in self._attrs:
            if hasattr(self, attr):
                model_dir += "/%s(%s)" % (attr, getattr(self, attr))
        return model_dir

    def save(self, checkpoint_dir, global_step=None):
        self.saver = tf.train.Saver()

        print("[*] Saving checkpoints...")
        model_name = type(self).__name__

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)


    def load(self, checkpoint_dir):
        self.saver = tf.train.Saver()

        print("[*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = tf.train.latest_checkpoint(checkpoint_dir)
            self.saver.restore(self.sess, ckpt_name)
            print("[*] Loading path: {} ...".format(ckpt_name))
            print("[*] Load SUCCESS")
            return True
        else:
            print("[!] Load failed...")
            return False