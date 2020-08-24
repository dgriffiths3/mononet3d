import os
import sys
import time

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import toml
import numpy as np
import pylab as plt
import pyvista as pv
import tensorflow as tf

from model import MonoNet
from utils import helpers, losses, tf_utils
from utils.dataset import load_dataset

tf.random.set_seed(123)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def plot_kitti(scan, img, pred_c, pred_attr, pred_clf, label):

	cmap = helpers.colormap(cfg['model']['n_classes'])

	pred_oabb = helpers.kitti_scenenet_to_oabb(pred_c, pred_attr)

	label_mask = tf.cast(label['bbox_3d'], tf.bool)[:, :, 0, 0]
	label_oabb = tf.boolean_mask(label['bbox_3d'], label_mask).numpy()

	plot = pv.Plotter(off_screen=not VIS, shape=(2, 1), window_size=(1240, 800))

	plot.subplot(0, 0)
	plot.add_points(scan, scalars=scan[:, 2], render_points_as_spheres=True)

	for i, box in enumerate(pred_oabb):
		for l in helpers.make_lines(box):
			plot.add_mesh(l, color=cmap[pred_clf[i][0]], line_width=3)

	for i, box in enumerate(label_oabb):
		for l in helpers.make_lines(box):
			plot.add_mesh(l, color=[0, 1, 0], line_width=3)

	plot.view_xz()

	plot.subplot(1, 0)
	img = np.flip(img.transpose(1, 0, 2), 1)
	cpos = [
		(img.shape[0]/2., img.shape[1]/2., 770),
		(img.shape[0]/2., img.shape[1]/2., 0.0),
		(0.0, 1.0, 0.0)
	]
	plot.camera_position = cpos
	plot.add_mesh(img[:, :, None, 0], scalars=img.reshape((-1, 3), order='F'), rgb=True)
	plot.remove_scalar_bar()

	if VIS:
		plot.show()
	else:
		if not os.path.isdir(SCREENSHOT_SAVE_DIR): os.makedirs(SCREENSHOT_SAVE_DIR)
		plot.show(screenshot=os.path.join(SCREENSHOT_SAVE_DIR, 'pred_{}'.format(int(time.time()))))


def inference():

	model = MonoNet(cfg['model'])
	model.load_weights(WEIGHTS)
	print('[info] model weights loaded.')

	dataset = load_dataset(DATASET, cfg, False)

	for inputs, label in dataset:

		calib = inputs['calib']

		pred_c, pred_attr, pred_clf = model([inputs['img'], calib['calib'], calib['img_orig']])

		img = tf.squeeze(inputs['img'], 0)
		scan = tf.squeeze(inputs['scan'], 0)
		pred_c = tf.squeeze(pred_c, 0)
		pred_attr = tf.squeeze(pred_attr, 0)
		pred_clf = tf.squeeze(pred_clf, 0)

		scores = tf.expand_dims(tf.reduce_max(pred_clf, 1), -1)
		pred_clf = tf.expand_dims(tf.math.argmax(pred_clf, 1), -1)
		scores = tf.where(pred_clf!=cfg['model']['n_classes']-1, scores, tf.zeros_like(scores))

		pred_c, pred_attr, pred_clf, scores = tf_utils.objectness_mask(
			pred_c, pred_attr, pred_clf, scores, SCORE_THRESH
		)
		
		if pred_c.shape[0] == 0: continue

		if NMS:
			boxes = tf.squeeze(tf_utils.scenenet_to_aabb(
				tf.expand_dims(pred_c, 0), tf.expand_dims(pred_attr, 0)
			), 0)
			nms_inds = helpers.nms(boxes.numpy(), scores.numpy(), cfg['std']['max_labels'], NMS_THRESH)
			pred_c = tf.gather(pred_c, nms_inds)
			pred_attr = tf.gather(pred_attr, nms_inds)
			pred_clf = tf.gather(pred_clf, nms_inds)
			pred_boxes = tf.gather(boxes, nms_inds)
		else:
			pred_boxes = tf.squeeze(tf_utils.scenenet_to_aabb(
				tf.expand_dims(pred_c, 0), tf.expand_dims(pred_attr, 0)
			), 0)

		plot_kitti(
			scan.numpy(),
			img.numpy(),
			pred_c.numpy(),
			pred_attr.numpy(),
			pred_clf.numpy(),
			label
		)


if __name__ == '__main__':

	LOG_DIR = './logs/kitti_car_1'
	MODEL_DIR = 'model'
	DATASET = './data/kitti_car_val.tfrecord'
	WEIGHTS = os.path.join(LOG_DIR, MODEL_DIR, 'weights.ckpt')
	SCORE_THRESH = 0.5
	NMS_THRESH = 0.25
	NMS = False
	VIS = True
	SCREENSHOT_SAVE_DIR = './data/screenshots'

	cfg = toml.load(os.path.join(LOG_DIR, 'config.toml'))
	cfg['model']['batch_size'] = 1

	inference()
