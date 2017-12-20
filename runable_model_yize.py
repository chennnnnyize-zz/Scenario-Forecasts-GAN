#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#import ipdb

#batchnormalize:看
def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b

    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b

    else:
        raise NotImplementedError

    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=o, logits=t))

class DCGAN():
    def __init__(
            self,
            batch_size=100,
            image_shape=[24,24,1],
            dim_z=100,
            dim_y=5,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            lam=0.05
            ):

        self.lam=lam
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = tf.Variable(tf.random_normal([dim_z+dim_y, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(tf.random_normal([dim_W1+dim_y, dim_W2*6*6], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(tf.random_normal([5,5,dim_W3,dim_W2+dim_y], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(tf.random_normal([5,5,dim_channel,dim_W3+dim_y], stddev=0.02), name='gen_W4')

        self.discrim_W1 = tf.Variable(tf.random_normal([5,5,dim_channel+dim_y,dim_W3], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(tf.random_normal([5,5,dim_W3+dim_y,dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(tf.random_normal([dim_W2*6*6+dim_y,dim_W1], stddev=0.02), name='discrim_W3')
        self.discrim_W4 = tf.Variable(tf.random_normal([dim_W1+dim_y,1], stddev=0.02), name='discrim_W4')



    def build_model(self):
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])

        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        pred_high = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        pred_low = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        h4 = self.generate(Z, Y)
        #image_gen comes from sigmoid output of generator
        image_gen = tf.nn.sigmoid(h4)

        raw_real2 = self.discriminate(image_real, Y)
        #p_real = tf.nn.sigmoid(raw_real)
        p_real = tf.reduce_mean(raw_real2)

        raw_gen2 = self.discriminate(image_gen, Y)
        #p_gen = tf.nn.sigmoid(raw_gen)
        p_gen = tf.reduce_mean(raw_gen2)


        discrim_cost = tf.reduce_mean(raw_real2) - tf.reduce_mean(raw_gen2)
        gen_cost = -tf.reduce_mean(raw_gen2)

        mask = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='mask')
        '''contextual_loss_latter = tf.reduce_sum(tf.contrib.layers.flatten(
            -tf.log(tf.abs(image_real-image_gen))), 1)'''
        #contextual_loss_latter = tf.reduce_sum(tf.log(tf.contrib.layers.flatten(tf.abs(image_gen - pred_high))), 1)

        #log loss
        '''contextual_loss_latter = tf.reduce_sum(tf.contrib.layers.flatten(
        -tf.log(tf.maximum(
            (mask + tf.multiply(tf.ones_like(mask) - mask, pred_high)) - tf.multiply(
                tf.ones_like(mask) - mask, image_gen), 0.0001*tf.ones_like(mask)))
        -tf.log(tf.maximum(
            (mask + tf.multiply(tf.ones_like(mask) - mask, image_gen)) - tf.multiply(
                tf.ones_like(mask) - mask, pred_low), 0.0001*tf.ones_like(mask)))), 1)'''
        contextual_loss_latter = tf.contrib.layers.flatten(
            -tf.log(
                (mask + tf.multiply(tf.ones_like(mask) - mask, pred_high)) - tf.multiply(
                    tf.ones_like(mask) - mask, image_gen))
            - tf.log(
                (mask + tf.multiply(tf.ones_like(mask) - mask, image_gen)) - tf.multiply(
                    tf.ones_like(mask) - mask, pred_low)))
        contextual_loss_latter = tf.where(tf.is_nan(contextual_loss_latter), tf.ones_like(contextual_loss_latter) * 1000000.0, contextual_loss_latter)
        contextual_loss_latter2 = tf.reduce_sum(contextual_loss_latter, 1)
        #square loss
        '''contextual_loss_latter = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(tf.ones_like(mask) - mask, image_gen) - tf.multiply(tf.ones_like(mask) - mask, pred_high)))
        +tf.contrib.layers.flatten(
            tf.square(
                tf.multiply(tf.ones_like(mask) - mask, image_gen) - tf.multiply(tf.ones_like(mask) - mask, pred_high)))
        , 1)'''
        contextual_loss_former = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(mask, image_gen) - tf.multiply(mask, image_real))), 1)
        contextual_loss_prepare = tf.reduce_sum(tf.contrib.layers.flatten(
            tf.square(tf.multiply(tf.ones_like(mask) - mask, image_gen) - tf.multiply(tf.ones_like(mask)-mask, image_real))), 1)
        perceptual_loss = gen_cost
        complete_loss = contextual_loss_former + self.lam * perceptual_loss + 0.05*contextual_loss_latter2
        grad_complete_loss = tf.gradients(complete_loss, Z)
        grad_uniform_loss = tf.gradients(contextual_loss_prepare, Z)

        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen, grad_complete_loss, \
               pred_high, pred_low, mask, contextual_loss_latter, contextual_loss_former, grad_uniform_loss


    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.stack([self.batch_size, 1, 1, self.dim_y]))
        X = tf.concat([image, yb * tf.ones([self.batch_size, 24, 24, self.dim_y])],3)

        h1 = lrelu( tf.nn.conv2d( X, self.discrim_W1, strides=[1,2,2,1], padding='SAME' ))
        h1 = tf.concat([h1, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])],3)

        h2 = lrelu(batchnormalize( tf.nn.conv2d( h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')) )
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat([h2, Y], 1)
        discri=tf.matmul(h2, self.discrim_W3 )
        h3 = lrelu(batchnormalize(discri))
        return h3


    def generate(self, Z, Y):

        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat([Z,Y],1)
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.concat([h1, Y],1)
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size,6,6,self.dim_W2])
        h2 = tf.concat([h2, yb*tf.ones([self.batch_size, 6,6, self.dim_y])],3)

        output_shape_l3 = [self.batch_size,12,12,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat([h3, yb*tf.ones([self.batch_size, 12, 12, self.dim_y])], 3)

        output_shape_l4 = [self.batch_size,24,24,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        return h4


    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat([Z,Y], 1)
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat([h1, Y], 1)
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size,6, 6,self.dim_W2])
        h2 = tf.concat([h2, yb*tf.ones([batch_size, 6,6, self.dim_y])], 3)

        output_shape_l3 = [batch_size,12, 12,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat([h3, yb*tf.ones([batch_size, 12,12,self.dim_y])], 3)

        output_shape_l4 = [batch_size,24, 24,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        x = tf.nn.sigmoid(h4)
        return Z,Y,x
