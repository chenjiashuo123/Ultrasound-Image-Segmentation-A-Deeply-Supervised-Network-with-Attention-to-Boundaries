import time
import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from losses import sigmoid_cross_entropy_balanced

class DSN():

    def __init__(self, cfgs, run='training'):
        #配置文件参数
        self.cfgs = cfgs

        self.define_model()
    def define_model(self):
        """
        网络模型建立
        :return:
        """
        start_time = time.time()

        self.conv1_1

    def conv_layer_dsn(self,botom,name):
        """
        添加一个卷积层和权重参数
        :param botom:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            filt = self.ge
            tf.glorot_normal_initializer
            xavier_initializer()