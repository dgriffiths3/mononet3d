import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2, VGG16, VGG19

from utils.encoders import AlexNet
from utils import tf_utils, losses
from utils.halton import halton_seq

import time
import cv2 as cv
import numpy as np
import pylab as plt

import pyvista as pv


class MonoNet(keras.Model):


	def __init__(self, cfg):
		super(MonoNet, self).__init__()

		self.batch_size = cfg['batch_size']
		self.n_pred = cfg['n_pred']
		self.kernel_initializer = cfg['kernel_initializer']
		self.bias = cfg['bias']
		self.patch_size = cfg['patch_size']
		self.clf_dist = cfg['clf_dist']
		self.img_size = cfg['img_size']

		self.n_classes = cfg['n_classes']

		self.activation = tf.nn.leaky_relu

		ext = cfg['halton']
		self.halton_offset = halton_seq(self.batch_size, 3, self.n_pred, ext)

		self.define_network()


	def compile(self, s_optimizer, m_optimizer, eager=False):
		super(MonoNet, self).compile(run_eagerly=eager)

		self.optimizer = s_optimizer
		self.s_optimizer = s_optimizer
		self.m_optimizer = m_optimizer

		metrics = ['loss', 'center', 'attr', 'clf']
		self.train_metrics = [keras.metrics.Mean(name=m) for m in metrics]
		self.val_metrics = [keras.metrics.Mean(name=m) for m in metrics[:4]]


	def define_network(self):

		latent_dim = 512

		self.img_encoder = VGG16(
			include_top=False,
			weights='imagenet',
			input_shape=(*self.img_size[:2], 3),
			pooling='max'
		)

		self.mininet = AlexNet(self.patch_size)

		self.center_mlp = keras.Sequential([
			keras.Input(shape=(latent_dim,)),
			layers.Dense(self.n_pred*512, self.activation, self.bias, self.kernel_initializer),
			layers.Reshape([self.n_pred, 512]),
			layers.Dense(256, self.activation, self.bias, self.kernel_initializer),
			layers.Dense(3, None, self.bias, self.kernel_initializer),
			layers.Lambda(lambda x : x + self.halton_offset)
		], name='center_mlp')

		self.attr_mlp = keras.Sequential([
			keras.Input(shape=(latent_dim,)),
			layers.Dense(64, self.activation, self.bias, self.kernel_initializer),
			layers.Dense(32, self.activation, self.bias, self.kernel_initializer),
			layers.Dense(5, None, self.bias, self.kernel_initializer)
		], name='attr_mlp')

		self.clf_mlp = keras.Sequential([
			keras.Input(shape=(latent_dim,)),
			layers.Dense(64, self.activation, self.bias, self.kernel_initializer),
			layers.Dense(32, self.activation, self.bias, self.kernel_initializer),
			layers.Dense(self.n_classes, tf.nn.softmax, self.bias, self.kernel_initializer)
		], name='clf_mlp')


	def update_metrics(self, updates, set):

		metrics = self.train_metrics if set == 'train' else self.val_metrics
		[m.update_state([x]) for m, x in zip(metrics, updates)]
		

	def forward_pass(self, input, calib, o_img_size):

		scene_code = self.img_encoder(input)

		c_3d = self.center_mlp(scene_code)

		patches = tf_utils.obj_to_cam_crop(c_3d, input, self.patch_size, calib, o_img_size)
		patch_code = self.mininet(patches)

		attr = self.attr_mlp(patch_code)
		attr = tf.reshape(attr, (*c_3d.shape[:2], -1))

		clf = self.clf_mlp(patch_code)
		clf = tf.reshape(clf, (*c_3d.shape[:2], self.n_classes))

		return c_3d, attr, clf


	def supervised_loss(self, label, pred_c, pred_attr, pred_clf):

		loss_s, dists, idx = losses.chamfer_loss(label['c_3d'], pred_c)
		loss_attr = losses.attr_mse_loss(label['attr'], pred_attr, dists, idx, self.clf_dist)

		if self.optimizer.iterations > 100:
			loss_clf = losses.xe_loss(label['clf'], pred_clf, dists, idx, self.clf_dist, self.n_classes)
		else:
			loss_clf = 0.

		loss_m = loss_attr + loss_clf

		return loss_s, loss_m, loss_attr, loss_clf


	def train_step(self, input):

		label = input[-1]
		calib = input[0]['calib']

		with tf.GradientTape(persistent=True) as tape:

			pred_c, pred_attr, pred_clf = self.forward_pass(input[0]['img'], calib['calib'], calib['img_orig'])

			loss_s, loss_m, loss_attr, loss_clf = self.supervised_loss(
				label, pred_c, pred_attr, pred_clf)

		# Update scene net
		train_vars = self.img_encoder.trainable_variables + self.center_mlp.trainable_variables
		gradients = tape.gradient(loss_s, train_vars)
		self.s_optimizer.apply_gradients(zip(gradients, train_vars))

		# Update mini net
		train_vars = self.mininet.trainable_variables + self.attr_mlp.trainable_variables + self.clf_mlp.trainable_variables
		gradients = tape.gradient(loss_m, train_vars)
		self.m_optimizer.apply_gradients(zip(gradients, train_vars))

		# Update metrics
		updates = [loss_s+loss_m, loss_s, loss_attr, loss_clf]
		self.update_metrics(updates, 'train')
			
		return {m.name: m.result() for m in self.train_metrics}


	def test_step(self, input):

		label = input[-1]
		calib = input[0]['calib']

		pred_c, pred_attr, pred_clf = self.forward_pass(input[0]['img'], calib['calib'], calib['img_orig'])

		loss_s, loss_m, loss_attr, loss_clf = self.supervised_loss(label, pred_c, pred_attr, pred_clf)

		updates = [loss_s+loss_m, loss_s, loss_attr, loss_clf]
		self.update_metrics(updates, 'val')

		return {m.name: m.result() for m in self.val_metrics}


	def call(self, input):

		return self.forward_pass(input[0], input[1], input[2])
