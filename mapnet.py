from tensorflow.keras import *
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, AveragePooling2D
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import BatchNormalization
from tensorflow.image import resize
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import os
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


def conv2d(input, filters, kernel_size=3, stride=1, padding='same'):
    return Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=stride,
                  use_bias=False, kernel_initializer=VarianceScaling())(input)

def bn(input):
    return BatchNormalization(momentum=0.99, epsilon=1e-3)(input)

def bottleneck(x, size, downsampe=False):
    residual = x
    out = bn(x)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 1, padding='valid')
    out = bn(out)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)
    out = bn(out)
    out = tf.nn.relu(out)
    out = conv2d(out, size * 4, 1, padding='valid')

    if downsampe:
        residual = bn(x)
        residual = tf.nn.relu(residual)
        residual = conv2d(residual, size * 4, 1, padding='valid')
    out = tf.add(out, residual)
    return out


def stage0(x):
    x = bottleneck(x, 64, downsampe=True)
    x = bottleneck(x, 64)
    x = bottleneck(x, 64)
    x = bottleneck(x, 64)
    return x


def translayer(x, in_channels, out_channels):
    num_in = len(in_channels)
    num_out = len(out_channels)
    out = []
    for i in range(num_out):
        if i < num_in:
            residual = bn(x[i])
            residual = tf.nn.relu(residual)
            residual = conv2d(residual, out_channels[i], 3)
            out.append(residual)
        else:
            residual = bn(x[-1])
            residual = tf.nn.relu(residual)
            residual = conv2d(residual, out_channels[i], 3, stride=2)
            out.append(residual)
    return out

def resblock(x, size):
    residual = x

    out = bn(x)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)
    out = bn(out)
    out = tf.nn.relu(out)
    out = conv2d(out, size, 3)

    out = tf.add(out, residual)
    return out

def convb(x, block_num, channels):
    out = []
    for i in range(len(channels)):
        residual = x[i]
        for j in range(block_num):
            residual = resblock(residual, channels[i])
        out.append(residual)
    return out

def featfuse(x, channels, multi_scale_output=True):
    out = []
    for i in range(len(channels) if multi_scale_output else 1):
        residual = x[i]
        for j in range(len(channels)):
            if j > i:
                if multi_scale_output == False:
                    y = bn(x[j])
                    y = tf.nn.relu(y)
                    y = conv2d(y, channels[j], 1, padding='VALID')
                    out.append(tf.keras.layers.UpSampling2D(size=2 ** (j - i))(y))
                else:
                    y = bn(x[j])
                    y = tf.nn.relu(y)
                    y = conv2d(y, channels[i], 1, padding='VALID')
                    y = tf.keras.layers.UpSampling2D(size=2 ** (j - i))(y)
                    residual = tf.add(residual, y)

            elif j < i:
                y = x[j]
                for k in range(i - j):
                    if k == i - j - 1:
                        y = bn(y)
                        y = tf.nn.relu(y)
                        y = conv2d(y, channels[i], 1)
                        y = MaxPooling2D(pool_size=(2, 2))(y)
                        # y = tf.layers.max_pooling2d(y, 2, 2)

                    else:
                        y = bn(y)
                        y = tf.nn.relu(y)
                        y = conv2d(y, channels[j], 1)
                        y = MaxPooling2D(pool_size=(2, 2))(y)
                        # y = tf.max_pooling2d(y, 2, 2)

                residual = tf.add(residual, y)
        out.append(residual)
    return out

def convblock(x, channels, multi_scale_output=True):
    residual = convb(x, 4, channels)
    out = featfuse(residual, channels, multi_scale_output=multi_scale_output)
    return out

def stage(x, num_modules, channels,multi_scale_output=True):
    out = x
    for i in range(num_modules):
        if i == num_modules - 1 and multi_scale_output == False:
            out = convblock(out, channels, multi_scale_output=False)
        else:
            out = convblock(out, channels)
    return out

def pyramid_pooling_block(input, bin_sizes):
    pool_list = []
    h = input.shape[1]
    c = input.shape[-1]
    for bin_size in bin_sizes:
        pool1 = AveragePooling2D((h // bin_size, h // bin_size), (h // bin_size, h // bin_size))
        # pool1 = tf.layers.average_pooling2d(input, (h // bin_size, h // bin_size), (h // bin_size, h // bin_size))

        pool1 = conv2d(pool1, int(c)//4, 1)

        pool1 = resize(pool1, (h,h))
        # pool1 = tf.image.resize_bilinear(pool1, (h, h))
        pool_list.append(pool1)
    pool = tf.concat(pool_list, axis=3)
    return tf.add(input, pool)

def spatial_pooling(input):
    h,w=input.shape[1],input.shape[2]
    p1 = resize(MaxPooling2D(pool_size=(2,2))(input), (h,w))
    # p1=tf.image.resize_bilinear(tf.layers.max_pooling2d(input,2,2),(h,w))

    p2 = resize(MaxPooling2D(pool_size=(3,3))(input), (h,w))
    # p2 = tf.image.resize_bilinear(tf.layers.max_pooling2d(input, 3, 3), (h, w))

    p3 = resize(MaxPooling2D(pool_size=(5,5))(input), (h,w))
    # p3=tf.image.resize_bilinear(tf.layers.max_pooling2d(input,5,5),(h,w))

    p4 = resize(MaxPooling2D(pool_size=(6,6))(input), (h,w))
    # p4 = tf.image.resize_bilinear(tf.layers.max_pooling2d(input, 6, 6), (h, w))

    p=tf.concat([p1,p2,p3,p4,input],axis=-1)
    return p


def channel_squeeze(input,filters,name=" "):
    with tf.name_scope(name):
        squeeze=tf.reduce_mean(input,axis=[1,2])
        with tf.name_scope(name+"fc1"):
            fc1 = Dense(use_bias=True,units=filters)(squeeze)
            # fc1=tf.layers.dense(squeeze,use_bias=True,units=filters)

            fc1=tf.nn.relu(fc1)
        with tf.name_scope(name+"fc2"):
            fc2 = Dense(use_bias=True,units=filters)(fc1)
            # fc2=tf.layers.dense(fc1,use_bias=True,units=filters)
            fc2=tf.nn.sigmoid(fc2)
        result=tf.reshape(fc2,[-1,1,1,filters])
        return input*result


def my_coef(y_true, y_pred):
    T = K.flatten(y_true)
    P = K.flatten(y_pred)

    intersection = K.sum(2 * T * P)

    dice = (intersection + 1.0) / (K.sum(T) + K.sum(P) + 1.0)

    return dice

def my_loss(y_true, y_pred):
    return 1 - my_coef(y_true, y_pred)


def mapnet(input_size=(256, 256, 3), is_training=True):
    # channels_s2 = [64, 128]
    # channels_s3 = [64, 128, 256]
    channels_s2 = [32, 64]
    channels_s3 = [32, 64, 128]
    num_modules_s2 = 2
    num_modules_s3 = 3

    # Emre ss
    inputs = Input(input_size)

    conv_1 = conv2d(inputs, 64, stride=2)
    conv_1 = bn(conv_1)
    conv_1 = tf.nn.relu(conv_1)
    conv_2 = conv2d(conv_1, 64)
    conv_2 = bn(conv_2)
    conv_2 = tf.nn.relu(conv_2)
    conv_3 = conv2d(conv_2, 64)
    conv_3 = bn(conv_3)
    conv_3 = tf.nn.relu(conv_3)

    conv_4 = MaxPooling2D(pool_size=(2,2))(conv_3)
    # conv_4 = tf.layers.max_pooling2d(conv_3, 2, 2)

    stage1 = stage0(conv_4)
    trans1 = translayer([stage1], [256], channels_s2)
    stage2 = stage(trans1, num_modules_s2, channels_s2)
    trans2 = translayer(stage2, channels_s2, channels_s3)
    stage3 = stage(trans2, num_modules_s3, channels_s3,multi_scale_output=False)

    stg3=tf.concat(stage3,axis=-1)
    squeeze=channel_squeeze(stg3, stg3.shape[-1], name="squeeze")

    spatial=tf.concat([stage3[0],stage3[1]],axis=-1)
    # spatial=pyramid_pooling_block(spatial, [1, 2, 4, 8])
    spatial=spatial_pooling(spatial)

    new_feature = tf.concat([spatial, squeeze], axis=-1)
    new_feature = bn(new_feature)
    new_feature = tf.nn.relu(new_feature)
    result=conv2d(new_feature, 128, 1, padding='SAME')

    up1 = tf.image.resize(result, size=(stage3[0].shape[1] * 2, stage3[0].shape[2] * 2))
    # up1=tf.image.resize_bilinear(result,size=(stage3[0].shape[1]*2,stage3[0].shape[2]*2))

    up1 = bn(up1)
    up1 = tf.nn.relu(up1)
    up1 = conv2d(up1, 64, 3)

    up2 = tf.image.resize(up1, size=(up1.shape[1] * 2, up1.shape[2] * 2))
    # up2 = tf.image.resize_bilinear(up1, size=(up1.shape[1]*2, up1.shape[2]*2))

    up2 = bn(up2)
    up2 = tf.nn.relu(up2)
    up2 = conv2d(up2, 32, 3)

    up2 = bn(up2)
    up2 = tf.nn.relu(up2)
    final = conv2d(up2, 1, 1, padding='valid')
    final = tf.nn.sigmoid(final)
    # final = result
    model = Model(inputs, final)

    # model.compile(optimizer=Adam(lr=0.001), loss=my_loss, metrics=[my_coef])
    model.compile(optimizer=Adam(lr=0.001), loss=my_loss, metrics=[my_coef])

    return model


# if __name__ == '__main__':
#     m = mapnet()
#     print(m.summary())
