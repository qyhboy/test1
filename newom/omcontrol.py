#!/usr/bin/python
from __future__ import division, print_function, absolute_import
import csv
import datetime
import pymysql
import os, time, random
import  xml.dom.minidom
from xml.dom.minidom import parse
import xml.dom.minidom
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from multiprocessing import Process, Queue
#orange
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'  #No logging TF

import tensorflow as tf
import numpy as np
import time
from Generator import OmniglotGenerator
from numpy import *;
import random

other_class=0
nb_class =10+other_class
input_size = 784
img_size=(28,28)
batch_size_om = 16
nb_samples_per_class = 17
train_samples_per_class=1

train_samples=17
mb_size = 64

generator1 = OmniglotGenerator(data_folder='/home/pt/test1/data/omtrain', \
                               batch_size=batch_size_om, nb_samples=nb_class,\
                               nb_samples_per_class=nb_samples_per_class,\
                               max_rotation=0., max_shift=0.,img_size=img_size, max_iter=None,\
                               train_samples_per_class=train_samples_per_class)
x_test,y_test=generator1.sample(nb_class)

example_outputs = y_test.reshape(batch_size_om*nb_class*nb_samples_per_class,1)
example_input=x_test.reshape(batch_size_om*nb_class*nb_samples_per_class,784)

#test_others_input = np.zeros((batch_size_om *nb_samples_per_class*other_class,input_size))
#test_others_outputs  =np.zeros((batch_size_om *nb_samples_per_class*other_class,nb_class))

train_input=np.zeros((batch_size_om *train_samples*(nb_class-other_class),input_size))
train_outputs=np.zeros((batch_size_om *train_samples*(nb_class-other_class),nb_class))
jte=0
jtr=0
jto=0
for k in range (nb_class):
    for i in range (batch_size_om *nb_samples_per_class*nb_class):
        if example_outputs[i]==k:
            train_input[jtr]=example_input[i]
            train_outputs[jtr][k]=1
            jtr+=1
            
                

    
    
def getbatchtrain(size):
    batchtrain=np.zeros((size,input_size))
    labeltrain=np.zeros((size,nb_class))
    for i in range(size):
        idx_1 = random.randint(0, batch_size_om *(nb_class-other_class)*train_samples-1)
        batchtrain[i,:] = train_input[idx_1,:]
        labeltrain[i,:]=train_outputs[idx_1,:]
    return batchtrain,labeltrain

def getbatchtrain2(size):
    batchtrain=np.zeros((size,input_size))
    labeltrain=np.zeros((size,nb_class))
    batchtrain2=np.zeros((size,input_size))
    labeltrain2=np.zeros((size,nb_class))
    for i in range(size):
        idx_1 = random.randint(0, batch_size_om *(nb_class-other_class)*train_samples-1)
        jjj=idx_1/(batch_size_om*train_samples)
        idx_2 = random.randint(int(jjj)*(batch_size_om*train_samples),int(jjj+1)*(batch_size_om*train_samples)-1)
        batchtrain[i,:] = train_input[idx_1,:]
        labeltrain[i,:]=train_outputs[idx_1,:]
        batchtrain2[i,:] = train_input[idx_2,:]
        labeltrain2[i,:]=train_outputs[idx_2,:]
    return batchtrain,labeltrain,batchtrain2,labeltrain2

z,_=getbatchtrain(64)
#z,_=getbatchone(10)
z.shape


#CVAE2

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/home/pt/test1/ELM/data", one_hot=True)
# Parameters
learning_rate2 = 0.001
num_steps2 =1000000
batch_size2 = 64
y_dim2=784
# Network Parameters
image_dim2= 784 # MNIST images are 28x28 pixels
hidden_dim2 = 512
hidden_dim22 = 512
latent_dim2 =2

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights2 = {
    'encoder_h1': tf.Variable(glorot_init([image_dim2+y_dim2, hidden_dim2])),
    'z_mean': tf.Variable(glorot_init([hidden_dim2, latent_dim2])),
    'z_std': tf.Variable(glorot_init([hidden_dim2, latent_dim2])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim2+y_dim2, hidden_dim22])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim22, image_dim2]))
}
biases2 = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim2])),
    'z_mean': tf.Variable(glorot_init([latent_dim2])),
    'z_std': tf.Variable(glorot_init([latent_dim2])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim22])),
    'decoder_out': tf.Variable(glorot_init([image_dim2]))
}

# Building the encoder
input_image2 = tf.placeholder(tf.float32, shape=[None, image_dim2])


y2 = tf.placeholder(tf.float32, shape=[None, y_dim2])

encoder2 = tf.matmul(tf.concat([input_image2,y2], 1), weights2['encoder_h1']) + biases2['encoder_b1']
encoder2 = tf.nn.tanh(encoder2)
z_mean2 = tf.matmul(encoder2, weights2['z_mean']) + biases2['z_mean']
z_std2 = tf.matmul(encoder2, weights2['z_std']) + biases2['z_std']

# Sampler: Normal (gaussian) random distribution
eps2 = tf.random_normal(tf.shape(z_std2), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z2 = z_mean2 + tf.exp(z_std2 / 2) * eps2

# Building the decoder (with scope to re-use these layers later)
decoder2 = tf.matmul(tf.concat([z2,y2], 1), weights2['decoder_h1']) + biases2['decoder_b1']
decoder2 = tf.nn.tanh(decoder2)
decoder2 = tf.matmul(decoder2, weights2['decoder_out']) + biases2['decoder_out']
decoder2 = tf.nn.sigmoid(decoder2)

# Define VAE Loss
def vae_loss2(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std2 - tf.square(z_mean2) - tf.exp(z_std2)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op2 = vae_loss2(decoder2, input_image2)
optimizer2 = tf.train.RMSPropOptimizer(learning_rate=learning_rate2)
train_op2 = optimizer2.minimize(loss_op2)

# Initialize the variables (i.e. assign their default value)
#init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
#sess = tf.Session()

# Run the initializer
#sess.run(init)
#CVAE


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/home/pt/test1/ELM/data", one_hot=True)
# Parameters
learning_rate = 0.001
num_steps =80000
batch_size = 64
y_dim=10
# Network Parameters
image_dim = 784 # MNIST images are 28x28 pixels
hidden_dim = 512
latent_dim = 2

# A custom initialization (see Xavier Glorot init)
def glorot_init(shape):
    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))

# Variables
weights = {
    'encoder_h1': tf.Variable(glorot_init([image_dim+y_dim, hidden_dim])),
    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),
    'decoder_h1': tf.Variable(glorot_init([latent_dim+y_dim, hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))
}
biases = {
    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'z_mean': tf.Variable(glorot_init([latent_dim])),
    'z_std': tf.Variable(glorot_init([latent_dim])),
    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),
    'decoder_out': tf.Variable(glorot_init([image_dim]))
}

# Building the encoder
input_image = tf.placeholder(tf.float32, shape=[None, image_dim])
noise_VAE=tf.placeholder(tf.float32, shape=[None, image_dim])

y = tf.placeholder(tf.float32, shape=[None, y_dim])

encoder = tf.matmul(tf.concat([input_image,y], 1), weights['encoder_h1']) + biases['encoder_b1']
encoder = tf.nn.tanh(encoder)
z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']
z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']

# Sampler: Normal (gaussian) random distribution
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,
                       name='epsilon')
z = z_mean + tf.exp(z_std / 2) * eps

# Building the decoder (with scope to re-use these layers later)
decoder = tf.matmul(tf.concat([z,y], 1), weights['decoder_h1']) + biases['decoder_b1']
decoder = tf.nn.tanh(decoder)
decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']
decoder = tf.nn.sigmoid(decoder)

# Define VAE Loss
def vae_loss(x_reconstructed, x_true):
    # Reconstruction loss
    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \
                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)
    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)
    # KL Divergence loss
    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)
    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)
    return tf.reduce_mean(encode_decode_loss + kl_div_loss)

loss_op = vae_loss(decoder, input_image)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
#init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
#sess = tf.Session()

# Run the initializer
#sess.run(init)
sess = tf.Session()
saver = tf.train.Saver() 
sess.run(tf.global_variables_initializer())
saver.restore(sess, "/home/pt/test1/omnist2/newom/goodom1/drawmodel.ckpt")

import tensorflow as tf
import numpy as np
#import pytest

import os
import random
from Images import get_shuffled_images, time_offset_label, load_transform


def startom():   
    noise_input2 = tf.placeholder(tf.float32, shape=[None, latent_dim2])
    decoder2 = tf.matmul(tf.concat([noise_input2,y2], 1), weights2['decoder_h1']) + biases2['decoder_b1']
    decoder2 = tf.nn.tanh(decoder2)
    decoder2 = tf.matmul(decoder2, weights2['decoder_out']) + biases2['decoder_out']
    decoder2 = tf.nn.sigmoid(decoder2)
    n = 9
    noise_dim=2
    filename='/var/www/html/search/images/input.png'
    example_inputs = np.asarray([load_transform(filename, size=(28,28)).flatten() \
                                           ], dtype=np.float32)


    canvas_recon0 = np.empty((28 * (1), 28 * 1))
    canvas_recon1 = np.empty((28 * (1), 28 * 1))
    canvas_recon2 = np.empty((28 * (1), 28 * 1))
    canvas_recon3 = np.empty((28 * (1), 28 * 1))
    canvas_recon4 = np.empty((28 * (1), 28 * 1))
    canvas_recon5 = np.empty((28 * (1), 28 * 1))
    canvas_recon6 = np.empty((28 * (1), 28 * 1))
    canvas_recon7 = np.empty((28 * (1), 28 * 1))
    canvas_recon8 = np.empty((28 * (1), 28 * 1))


    for i in range(n):
        ztest = np.random.uniform(-1., 1., size=[1, noise_dim])
        g = sess.run(decoder2, feed_dict={noise_input2: ztest,y2:example_inputs })
        g = (g + 1.) / 2.
        g = -1 * (g - 1)
        if i==0:
            canvas_recon0[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon0, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==1:
            canvas_recon1[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon1, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==2:
            canvas_recon2[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon2, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==3:
            canvas_recon3[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon3, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==4:
            canvas_recon4[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon4, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==5:
            canvas_recon5[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon5, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==6:
            canvas_recon6[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon6, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==7:
            canvas_recon7[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon7, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )
        if i==8:
            canvas_recon8[ 0:(1) * 28,  0:(1) * 28] = g.reshape([28, 28])
            plt.figure(figsize=(1, 1))
            plt.axis('off')
            plt.imshow(canvas_recon8, origin="upper", cmap="gray")
            plt.savefig('/var/www/html/search/images/out%d.png'%(i+1) )


def begin():
    success=1
    while(success):
        time.sleep(random.random()*1)
        try:
            conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='pt123456', db='image', charset='utf8')
            cur=conn.cursor()
            sql="select id from  start " #0 not run;1 buildtable;2 sphinxfile--which times,
            aa=cur.execute(sql)
            info = cur.fetchmany(aa)
            for ii in info:
                a=ii[0]
            success=0
            if(a==0):
                success=1
            cur.close()
            conn.close()
        except (MySQLdb.Error,e): 
            tt=open('wrong.txt','w+')
            ff= 'Mysql Error %s: %s' % (e.args[0], e.args[-1])
            tt.write("wrong"+ff+"\n")
            tt.close()
    if (a==1):
        startom()
        done()

def done():
    try:
        conn = pymysql.connect(host='localhost', port=3306, user='root', passwd='pt123456', db='image', charset='utf8')
        cur=conn.cursor()
        sql="update start set id=0"
        cur.execute(sql)
        conn.commit()
        cur.close()
        conn.close()
    except (MySQLdb.Error,e): 
        tt=open('wrong.txt','a+')
        ff= 'Mysql Error %s: %s' % (e.args[0], e.args[-1])
        tt.write("wrong"+ff+"\n")
        tt.close()

def main():
    while(1):
        begin()

        
if __name__=='__main__':
    main()
    
