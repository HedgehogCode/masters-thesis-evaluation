# Translated to TF2 from https://github.com/LIAGM/LFattNet/blob/master/LFattNet_func/func_model_81.py
# Original Author: LIAGM <louis19950117@yahoo.com.tw>, License: MIT

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import Conv2D, Conv3D, AveragePooling2D, Lambda, GlobalAveragePooling3D
from tensorflow.keras.layers import BatchNormalization

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import backend as K
import numpy as np


def convbn(input, out_planes, kernel_size, stride, dilation):

    seq = Conv2D(out_planes, kernel_size, stride, 'same', dilation_rate=dilation, use_bias=False)(input)
    seq = BatchNormalization()(seq)

    return seq

def convbn_3d(input, out_planes, kernel_size, stride):
    seq = Conv3D(out_planes, kernel_size, stride, 'same', use_bias=False)(input)
    seq = BatchNormalization()(seq)

    return seq

def BasicBlock(input, planes, stride, downsample, dilation):
    conv1 = convbn(input, planes, 3, stride, dilation)
    conv1 = Activation('relu')(conv1)
    conv2 = convbn(conv1, planes, 3, 1, dilation)
    if downsample is not None:
        input = downsample

    conv2 = tf.keras.layers.add([conv2, input])
    return conv2


def _make_layer(input, planes, blocks, stride, dilation):
    inplanes = 4
    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = Conv2D(planes, 1, stride, 'same', use_bias=False)(input)
        downsample = BatchNormalization()(downsample)

    layers = BasicBlock(input, planes, stride, downsample, dilation)
    for i in range(1, blocks):
        layers = BasicBlock(layers, planes, 1, None, dilation)

    return layers

def UpSampling2DBilinear(size):
    def apply(x):
        return tf.image.resize(x, size, method='bilinear')
    return apply

def feature_extraction(sz_input, sz_input2):
    i = Input(shape=(sz_input, sz_input2, 1))
    firstconv = convbn(i, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)
    firstconv = convbn(firstconv, 4, 3, 1, 1)
    firstconv = Activation('relu')(firstconv)

    layer1 = _make_layer(firstconv, 4, 2, 1, 1)
    layer2 = _make_layer(layer1, 8, 8, 1, 1)
    layer3 = _make_layer(layer2, 16, 2, 1, 1)
    layer4 = _make_layer(layer3, 16, 2, 1, 2)

    # layer4_size = (layer4.get_shape().as_list()[1], layer4.get_shape().as_list()[2])
    layer4_size = tf.shape(layer4)[1:3]

    branch1 = AveragePooling2D((2, 2), (2, 2), 'same')(layer4)
    branch1 = convbn(branch1, 4, 1, 1, 1)
    branch1 = Activation('relu')(branch1)
    branch1 = UpSampling2DBilinear(layer4_size)(branch1)

    branch2 = AveragePooling2D((4, 4), (4, 4), 'same')(layer4)
    branch2 = convbn(branch2, 4, 1, 1, 1)
    branch2 = Activation('relu')(branch2)
    branch2 = UpSampling2DBilinear(layer4_size)(branch2)

    branch3 = AveragePooling2D((8, 8), (8, 8), 'same')(layer4)
    branch3 = convbn(branch3, 4, 1, 1, 1)
    branch3 = Activation('relu')(branch3)
    branch3 = UpSampling2DBilinear(layer4_size)(branch3)

    branch4 = AveragePooling2D((16, 16), (16, 16), 'same')(layer4)
    branch4 = convbn(branch4, 4, 1, 1, 1)
    branch4 = Activation('relu')(branch4)
    branch4 = UpSampling2DBilinear(layer4_size)(branch4)

    output_feature = tf.keras.layers.concatenate([layer2, layer4, branch4, branch3, branch2, branch1])
    lastconv = convbn(output_feature, 16, 3, 1, 1)
    lastconv = Activation('relu')(lastconv)
    lastconv = Conv2D(4, 1, (1, 1), 'same', use_bias=False)(lastconv)

    model = Model(inputs=[i], outputs=[lastconv])

    return model

def _getCostVolume_(inputs):
    shape = K.shape(inputs[0])
    disparity_costs = []
    for d in range(-4, 5):
        if d == 0:
            tmp_list = []
            for i in range(len(inputs)):
                tmp_list.append(inputs[i])
        else:
            tmp_list = []
            for i in range(len(inputs)):
                (v, u) = divmod(i, 9)
                translations = tf.constant([d * (u - 4), d * (v - 4)], dtype=tf.float32)
                tensor = tfa.image.translate(inputs[i], translations, 'BILINEAR')
                tmp_list.append(tensor)

        cost = K.concatenate(tmp_list, axis=3)
        disparity_costs.append(cost)
    cost_volume = K.stack(disparity_costs, axis=1)
    cost_volume = K.reshape(cost_volume, (shape[0], 9, shape[1], shape[2], 4*81))
    return cost_volume

def channel_attention(cost_volume):
    x = GlobalAveragePooling3D()(cost_volume)
    x = Lambda(lambda y: K.expand_dims(K.expand_dims(K.expand_dims(y, 1), 1), 1))(x)
    x = Conv3D(170, 1, 1, 'same')(x)
    x = Activation('relu')(x)
    x = Conv3D(15, 1, 1, 'same')(x) # [B, 1, 1, 1, 15]
    x = Activation('sigmoid')(x)

    # 15 -> 25
    # 0  1  2  3  4
    #    5  6  7  8
    #       9 10 11
    #         12 13
    #            14
    #
    # 0  1  2  3  4  
    # 1  5  6  7  8
    # 2  6  9 10 11
    # 3  7 10 12 13
    # 4  8 11 13 14

    x = Lambda(lambda y: K.concatenate([y[:, :, :, :, 0:5], y[:, :, :, :, 1:2], y[:, :, :, :, 5:9], y[:, :, :, :, 2:3],
                                         y[:, :, :, :, 6:7], y[:, :, :, :, 9:12], y[:, :, :, :, 3:4], y[:, :, :, :, 7:8],
                                         y[:, :, :, :, 10:11], y[:, :, :, :, 12:14], y[:, :, :, :, 4:5], y[:, :, :, :, 8:9],
                                         y[:, :, :, :, 11:12], y[:, :, :, :, 13:15]], axis=-1))(x)

    x = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 5, 5)))(x)
    x = Lambda(lambda y: tf.pad(tensor=y, paddings=[[0, 0], [0, 4], [0, 4]], mode='REFLECT'))(x)
    attention = Lambda(lambda y: K.reshape(y, (K.shape(y)[0], 1, 1, 1, 81)))(x)
    x = Lambda(lambda y:K.repeat_elements(y, 4, -1))(attention)
    return tf.keras.layers.multiply([x, cost_volume]), attention

def basic(cost_volume):

    feature = 2*75
    dres0 = convbn_3d(cost_volume, feature, 3, 1)
    dres0 = Activation('relu')(dres0)
    dres0 = convbn_3d(dres0, feature, 3, 1)
    cost0 = Activation('relu')(dres0)

    dres1 = convbn_3d(cost0, feature, 3, 1)
    dres1 = Activation('relu')(dres1)
    dres1 = convbn_3d(dres1, feature, 3, 1)
    cost0 = tf.keras.layers.add([dres1, cost0])

    dres4 = convbn_3d(cost0, feature, 3, 1)
    dres4 = Activation('relu')(dres4)
    dres4 = convbn_3d(dres4, feature, 3, 1)
    cost0 = tf.keras.layers.add([dres4, cost0])

    classify = convbn_3d(cost0, feature, 3, 1)
    classify = Activation('relu')(classify)
    cost = Conv3D(1, 3, 1, 'same', use_bias=False)(classify)

    return cost

def disparityregression(input):
    shape = K.shape(input)
    disparity_values = np.linspace(-4, 4, 9)
    x = K.constant(disparity_values, shape=[9])
    x = K.expand_dims(K.expand_dims(K.expand_dims(x, 0), 0), 0)
    x = tf.tile(x, [shape[0], shape[1], shape[2], 1])
    out = K.sum(tf.keras.layers.multiply([input, x]), -1)
    return out


def define_LFattNet(sz_input, sz_input2, view_n, learning_rate):

    """ 81 inputs"""
    input_list = []
    for i in range(len(view_n)*len(view_n)):
        print('input '+str(i))
        input_list.append(Input(shape=(sz_input, sz_input2, 1)))

    """ 81 features"""
    feature_extraction_layer = feature_extraction(sz_input, sz_input2)

    feature_list = []
    for i in range(len(view_n)*len(view_n)):
        print('feature '+str(i))
        feature_list.append(feature_extraction_layer(input_list[i]))

    """ cost volume """
    cv = Lambda(_getCostVolume_)(feature_list)

    """ channel attention """
    cv, attention = channel_attention(cv)

    """ cost volume regression """
    cost = basic(cv)
    cost = Lambda(lambda x: K.permute_dimensions(K.squeeze(x, -1), (0, 2, 3, 1)))(cost)
    pred = Activation('softmax')(cost)

    pred = Lambda(disparityregression)(pred)

    # when training use below
    # model = Model(inputs=input_list, outputs=[pred])

    # when evaluation use below
    model = Model(inputs=input_list, outputs=[pred, attention])
    
    model.summary()

    opt = Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss='mae')

    return model