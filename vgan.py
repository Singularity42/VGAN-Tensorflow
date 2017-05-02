import tensorflow as tf
import numpy as np
import h5py
import os
from layers import *

z_size = 100
batch_size = 10 IS_TRAIN = True
lr = 0.0002
beta1 = 0.5
train_iters = 10000
A,B,C = 64,64,32
channel = 3
img_size = 64*64

model_file_name = "results/vgan_applyeyemakeup"
tf.reset_default_graph()

def next_batch(data_array):
    length=data_array.shape[0] #assuming the data array to be a np arry
    permutations=np.random.permutation(length)
    idxs=permutations[0:batch_size]
    batch=np.zeros([batch_size, C, img_size*channel], dtype=np.float32)
    for i in range(len(idxs)):
        batch[i,:]=data_array[idxs[i]].reshape(C,img_size*channel)
    return batch / 255.

def static_net(z):
	static_input = tf.reshape(z,[batch_size, 1, 1, z_size])
	
	static_dconv1 = deconv2d(static_input,[4,4,512,z_size],[batch_size, 4,4,512],[1,1,1,1],'VALID',name='st_deconv1')
	static_dconv1 = batch_norm(static_dconv1, scope='st_bn_dconv1', is_training=IS_TRAIN)
	static_dconv1 = tf.nn.relu(static_dconv1)

	static_dconv2 = deconv2d(static_dconv1,[4,4,256,512],[batch_size, 8,8,256],[1,2,2,1],'SAME',name='st_deconv2')
	static_dconv2 = batch_norm(static_dconv2, scope='st_bn_dconv2', is_training=IS_TRAIN)
	static_dconv2 = tf.nn.relu(static_dconv2)

	static_dconv3 = deconv2d(static_dconv2,[4,4,128,256],[batch_size, 16,16,128],[1,2,2,1],'SAME',name='st_deconv3')
	static_dconv3 = batch_norm(static_dconv3, scope='st_bn_dconv3', is_training=IS_TRAIN)
	static_dconv3 = tf.nn.relu(static_dconv3)

	static_dconv4 = deconv2d(static_dconv3,[4,4,64,128],[batch_size, 32, 32,64],[1,2,2,1],'SAME',name='st_deconv4')
	static_dconv4 = batch_norm(static_dconv4, scope='st_bn_dconv4', is_training=IS_TRAIN)
	static_dconv4 = tf.nn.relu(static_dconv4)

	static_dconv5 = deconv2d(static_dconv4,[4,4,3,64],[batch_size, 64,64, 3],[1,2,2,1],'SAME',name='st_deconv5')
	static_dconv5 = tf.nn.tanh(static_dconv5)

	return static_dconv5


def video_net_and_mask(z):
	video_input = tf.reshape(z,[batch_size, 1, 1, 1, z_size])

	video_dconv1 = deconv3d(video_input,[2,4,4,512,z_size],[batch_size, 2, 4,4,512],[1,1,1,1,1],'VALID',name='video_deconv1')
	video_dconv1 = batch_norm(video_dconv1, scope='vd_bn_dconv1', is_training=IS_TRAIN)
	video_dconv1 = tf.nn.relu(video_dconv1)

	video_dconv2 = deconv3d(video_dconv1,[4,4,4,256,512],[batch_size, 4, 8, 8, 256],[1,2,2,2,1],'SAME',name='video_deconv2')
	video_dconv2 = batch_norm(video_dconv2, scope='vd_bn_dconv2', is_training=IS_TRAIN)
	video_dconv2 = tf.nn.relu(video_dconv2)

	video_dconv3 = deconv3d(video_dconv2,[4,4,4,128,256],[batch_size, 8, 16, 16, 128],[1,2,2,2,1],'SAME',name='video_deconv3')
	video_dconv3 = batch_norm(video_dconv3, scope='vd_bn_dconv3', is_training=IS_TRAIN)
	video_dconv3 = tf.nn.relu(video_dconv3)

	video_dconv4 = deconv3d(video_dconv3,[4,4,4,64,128],[batch_size, 16, 32, 32, 64],[1,2,2,2,1],'SAME',name='video_deconv4')
	video_dconv4 = batch_norm(video_dconv4, scope='vd_bn_dconv4', is_training=IS_TRAIN)
	video_dconv4 = tf.nn.relu(video_dconv4)

	video_dconv5 = deconv3d(video_dconv4,[4,4,4,3,64],[batch_size, 32, 64, 64, 3],[1,2,2,2,1],'SAME',name='video_deconv5')
	video_dconv5 = tf.nn.tanh(video_dconv5)

	# mast out... (for the mast net)

	mask_deconv5, mask_deconv5_weights = deconv3d(video_dconv4,[4,4,4,1,64], [batch_size, 32, 64, 64, 1],[1,2,2,2,1],'SAME',name='mask_deconv5',with_w=True)
	mask_deconv5 = tf.nn.sigmoid(mask_deconv5)

	l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.1, scope='mask_l1')
	mask_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [mask_deconv5_weights])

	return video_dconv5,mask_deconv5,mask_penalty

def discriminator_net(input,reuse=None):
	disc_conv1 = conv3d(input,[4,4,4,3,64],[1,2,2,2,1],'SAME',name='dc_conv1',reuse=reuse)
	disc_conv1 = lrelu(disc_conv1)

	disc_conv2 = conv3d(disc_conv1,[4,4,4,64,128],[1,2,2,2,1],'SAME',name='dc_conv2',reuse=reuse)
	disc_conv2 = batch_norm(disc_conv2,eps=1e-3,scope='dc_bn_conv2', is_training=IS_TRAIN,reuse=reuse)
	disc_conv2 = lrelu(disc_conv2)

	disc_conv3 = conv3d(disc_conv2,[4,4,4,128,256],[1,2,2,2,1],'SAME',name='dc_conv3',reuse=reuse)
	disc_conv3 = batch_norm(disc_conv3,eps=1e-3,scope='dc_bn_conv3', is_training=IS_TRAIN,reuse=reuse)
	disc_conv3 = lrelu(disc_conv3)

	disc_conv4 = conv3d(disc_conv3,[4,4,4,256,512],[1,2,2,2,1],'SAME',name='dc_conv4',reuse=reuse)
	disc_conv4 = batch_norm(disc_conv4,eps=1e-3,scope='dc_bn_conv4', is_training=IS_TRAIN,reuse=reuse)
	disc_conv4 = lrelu(disc_conv4)

	disc_conv5 = conv3d(disc_conv4,[2,4,4,512,1],[1,1,1,1,1],'VALID',name='dc_conv5',reuse=reuse)

	final = tf.reshape(disc_conv5,[batch_size])

	return tf.nn.sigmoid(final),final


def gen_video(z):
	background = static_net(z)
	background = tf.tile(tf.reshape(background,[batch_size,1,64,64,3]),[1,32,1,1,1])

	foreground,mask,penalty = video_net_and_mask(z)
	mask = tf.tile(mask,[1,1,1,1,3])

	video = foreground*mask + background*(1-mask)
	return video,penalty


z = tf.placeholder(tf.float32,[batch_size,z_size],name='z')
x = tf.placeholder(tf.float32,[batch_size,32,64,64,3],name='x_real')

gen_videos,penalty = gen_video(z)

gen_D, gen_D_logits = discriminator_net(gen_videos)
real_D, real_D_logits = discriminator_net(x,reuse=True)

d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=real_D_logits, labels=tf.ones_like(real_D_logits)))

d_loss_gen = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_D_logits, labels=tf.zeros_like(gen_D_logits)))

g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_D_logits, labels=tf.ones_like(gen_D_logits)))

d_loss = d_loss_gen + d_loss_real
g_loss = g_loss + penalty

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'dc_' in var.name]
g_vars = [var for var in t_vars if var not in d_vars]

d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(g_loss, var_list=g_vars)


dataset_filename = 'dataset/path_name'

hf = h5py.File(dataset_filename,'r')
caps = hf.get('filenames')
total_count = caps.shape[0]
train_data = hf.get('video_data')[:total_count].reshape(-1,C, A*B*channel)
print "loaded"

fetches = []
fetches.extend([d_loss,g_loss,d_optim,g_optim])
d_lss = [0]*train_iters
g_lss = [0]*train_iters


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.InteractiveSession(config=config)

saver = tf.train.Saver()

tf.initialize_all_variables().run()

for i in range(train_iters):
	xtrain = next_batch(train_data)
	xtrain = xtrain.reshape(-1,C,64,64,channel)
	z_sample = np.random.normal(size=(batch_size,z_size))
	# print z_sample.shape
	feed_dict={x:xtrain,z:z_sample}
	results = sess.run(fetches,feed_dict)
	d_lss[i],g_lss[i],_,_=results

	if i%10==0:
            print("iter=%d : D_Loss: %f G_Loss: %f" % (i, d_lss[i], g_lss[i]))
        if (i+1)%100==0:
            ckpt_file=model_file_name+".ckpt"
            print("Model saved in file: %s" % saver.save(sess,ckpt_file))


gen_sample = sess.run(gen_videos,{z:z_sample})
gen_sample = np.array(gen_sample)

out_file=model_file_name+".npy"
np.save(out_file,[gen_sample,d_lss,g_lss])
print("Outputs saved in file: %s" % out_file)

ckpt_file=model_file_name+".ckpt"
print("Model saved in file: %s" % saver.save(sess,ckpt_file))

sess.close()


print('Done with V-GAN!')


