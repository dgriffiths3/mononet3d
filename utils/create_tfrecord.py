import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import glob
import cv2 as cv
import numpy as np
import tensorflow as tf

from utils import tf_utils, helpers
import kitti_utils


def create_example(img, scan, label):

	feature = {
		'image/img' : tf_utils.bytes_feature(img),
		'image/orig' : tf_utils.float_list_feature(label['orig'].reshape(-1, 1)),
		'image/calib' : tf_utils.float_list_feature(label['calib'].reshape(-1, 1)),
		'scan/points' : tf_utils.float_list_feature(scan[:, :3].reshape(-1, 1)),
		'label/clf' : tf_utils.int64_list_feature(label['clf'].reshape(-1, 1)),
		'label/c_3d' : tf_utils.float_list_feature(label['c_3d'].reshape(-1, 1)),
		'label/bbox_3d' : tf_utils.float_list_feature(label['bbox_3d'].reshape(-1, 1)),
		'label/c_2d' : tf_utils.float_list_feature(label['c_2d'].reshape(-1, 1)),
		'label/bbox_2d' : tf_utils.float_list_feature(label['bbox_2d'].reshape(-1, 1)),
		'label/extent' : tf_utils.float_list_feature(label['extent'].reshape(-1, 1)),
		'label/rotation_i' : tf_utils.float_list_feature(label['ri'].reshape(-1, 1)),
		'label/rotation_j' : tf_utils.float_list_feature(label['rj'].reshape(-1, 1)),
	}

	return tf.train.Example(features=tf.train.Features(feature=feature))


def create_records():

	dataset = ['training', 'testing']
	max_objects = cfg['max_objects']

	train = True if cfg['dataset'] == 'training' else False
	 
	img_dir = os.path.join(cfg['in_dir'], 'data_object_image_2', cfg['dataset'], 'image_2')
	scan_dir = os.path.join(cfg['in_dir'], 'data_object_velodyne', cfg['dataset'], 'velodyne')
	calib_dir = os.path.join(cfg['in_dir'], 'data_object_calib', cfg['dataset'], 'calib')
	
	label_dir = os.path.join(cfg['in_dir'], 'data_object_label_2', 'training', 'label_2')

	img_files = glob.glob(os.path.join(img_dir, '*.png'))

	n_scenes = cfg['n_scenes'] if cfg['n_scenes'] > 0 else len(img_files)
	bar = helpers.progbar(n_scenes)
	bar.start()

	with tf.io.TFRecordWriter(cfg['out_train']) as train_writer, tf.io.TFRecordWriter(cfg['out_val']) as val_writer:

		for scene_id, img_file in enumerate(img_files):

			if scene_id == n_scenes: break

			bar.update(scene_id)

			base = os.path.splitext(os.path.basename(img_file))[0]

			scan_file = os.path.join(scan_dir, base+'.bin')
			calib_file = os.path.join(calib_dir, base+'.txt')
			label_file = os.path.join(label_dir, base+'.txt')

			img_arr = kitti_utils.load_img(img_file)
			orig_img_size = img_arr.shape[:2]
			img_arr = cv.resize(img_arr, (cfg['img_size'][1], cfg['img_size'][0]), interpolation=cv.INTER_CUBIC)
			_, img = cv.imencode('.png', img_arr)
			img = img.tobytes()

			scan = kitti_utils.load_scan(scan_file)
			calib = kitti_utils.load_calib(calib_file)
			label_k = kitti_utils.load_label(label_file)

			if label_k.shape[0] == 0: continue

			scan = kitti_utils.scan_to_camera_coords(scan, calib)
			_, scan = helpers.get_fixed_pts(scan, cfg['n_points'])

			label = {}

			n_inst = label_k.shape[0]

			label['calib'] = calib['P2']
			label['orig'] = np.array(orig_img_size).astype(np.float32)

			label['clf'] = label_k[:, 0, None].astype(int)
			label['clf'] = np.pad(label['clf'], [[0, max_objects-n_inst],[0, 0]], 'constant', constant_values=8)
			
			label['c_3d'] = label_k[:, 11:14]
			label['c_3d'] = np.pad(label['c_3d'], [[0, max_objects-n_inst],[0, 0]])

			label['extent'] = label_k[:, 8:11]
			label['extent'] = np.pad(label['extent'], [[0, max_objects-n_inst],[0, 0]])

			label['bbox_3d'] = kitti_utils.extract_3d_boxes(label_k)
			
			c_2d, bbox_2d = kitti_utils.extract_2d_boxes(label['bbox_3d'], calib, orig_img_size)
			label['c_2d'] = np.pad(c_2d, [[0, max_objects-n_inst],[0, 0]]).astype(np.float32)
			label['bbox_2d'] = np.pad(bbox_2d, [[0, max_objects-n_inst],[0, 0]]).astype(np.float32)
			
			label['bbox_3d'] = np.concatenate(
				[label['bbox_3d'], np.zeros((max_objects-n_inst, 8, 3))]
			)

			ri = np.cos(label_k[:, 14])
			rj = np.sin(label_k[:, 14])
			label['ri'] = np.pad(ri.reshape(-1, 1), [[0, max_objects-n_inst], [0, 0]])
			label['rj'] = np.pad(rj.reshape(-1, 1), [[0, max_objects-n_inst], [0, 0]])

			label = kitti_utils.remove_dontcare(label)

			tf_example = create_example(img, scan, label)

			if scene_id % 5 != 0:
				train_writer.write(tf_example.SerializeToString())
			else:
				val_writer.write(tf_example.SerializeToString())
	

if __name__ == '__main__':

	cfg = {
		'in_dir' : '../datasets/kitti',
		'dataset' : 'training',
		'out_train' : './data/kitti_car_train.tfrecord',
		'out_val' : './data/kitti_car_val.tfrecord',
		'n_scenes' : 10,
		'n_points' : 32768,
		'img_size': (375, 1240),
		'max_objects' : 22,
	}

	create_records()
