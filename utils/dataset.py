import sys
import os

sys.path.insert(0, './')

from utils import helpers
import numpy as np
import tensorflow as tf


def load_dataset(in_file, cfg, repeat=True):

	assert os.path.isfile(in_file), '[error] dataset path not found'

	n_points = cfg['model']['n_points']
	max_labels = cfg['std']['max_labels']
	img_size = cfg['model']['img_size']
	batch_size = cfg['model']['batch_size']
	
	shuffle_buffer = 1000

	def _extract_fn(data_record):

		in_features = {
			'image/img' : tf.io.FixedLenFeature([], tf.string),
			'image/orig' : tf.io.FixedLenFeature([2], tf.float32),
			'image/calib' : tf.io.FixedLenFeature([4 * 4], tf.float32),
			'scan/points' : tf.io.FixedLenFeature([n_points * 3], tf.float32),
			'label/clf' : tf.io.FixedLenFeature([max_labels], tf.int64),
			'label/c_3d' : tf.io.FixedLenFeature([max_labels * 3], tf.float32),
			'label/bbox_3d': tf.io.FixedLenFeature([max_labels * 3 * 8], tf.float32),
			'label/c_2d': tf.io.FixedLenFeature([max_labels * 2], tf.float32),
			'label/bbox_2d': tf.io.FixedLenFeature([max_labels * 4], tf.float32),
			'label/extent' : tf.io.FixedLenFeature([max_labels * 3], tf.float32),
			'label/rotation_i': tf.io.FixedLenFeature([max_labels], tf.float32),
			'label/rotation_j': tf.io.FixedLenFeature([max_labels], tf.float32),
		}

		return tf.io.parse_single_example(data_record, in_features)

	def _preprocess_fn(sample):

		img = tf.cast(tf.io.decode_png(sample['image/img'], 3), tf.float32) / 255.
		img = tf.image.resize(img, (img_size[0], img_size[1]))

		scan = tf.reshape(sample['scan/points'], (n_points, 3))
		scan = tf.random.shuffle(scan)

		rotation = tf.stack([sample['label/rotation_i'], sample['label/rotation_j']], -1)

		extent =  tf.reshape(sample['label/extent'], (max_labels, 3))
		attr = tf.concat([extent, rotation], -1)

		label = {
			'clf' : tf.reshape(sample['label/clf'], (max_labels, 1)),
			'c_3d' : tf.reshape(sample['label/c_3d'], (max_labels, 3)),
			'bbox_3d' : tf.reshape(sample['label/bbox_3d'], (max_labels, 8, 3)),
			'c_2d' : tf.reshape(sample['label/c_2d'], (max_labels, 2)),
			'bbox_2d' : tf.reshape(sample['label/bbox_2d'], (max_labels, 4)),
			'attr' : attr,
		}

		calib = {
			'calib' : tf.reshape(sample['image/calib'], (4, 4)),
			'img_orig' : sample['image/orig'],
		}

		return {'img': img, 'scan': scan, 'calib': calib}, label

	dataset = tf.data.TFRecordDataset(in_file)
	dataset = dataset.shuffle(shuffle_buffer)
	dataset = dataset.map(_extract_fn)
	dataset = dataset.map(_preprocess_fn)
	dataset = dataset.batch(batch_size, drop_remainder=True)

	if repeat == True: dataset = dataset.repeat()

	return dataset
