import sys

sys.path.insert(0, './')

import os
import cv2 as cv
import numpy as np

from utils import helpers


# CLASS_MAP = {
# 	'Car' : 0, 'Van' : 0, 'Truck' : 0, 'Pedestrian' : 1,
# 	'Person_sitting' : 1, 'Cyclist' : 2, 'Tram' : 3,
# 	'Misc' : 3, 'DontCare' : 3,
# }

CLASS_MAP = {
	'Car' : 0, 'Van' : 0, 'Truck' : 0, 'Pedestrian' : 1,
	'Person_sitting' : 1, 'Cyclist' : 1, 'Tram' : 1,
	'Misc' : 1, 'DontCare' : 1,
}


def load_img(file):

	return cv.imread(file)


def load_scan(file):

	return np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:, :3]


def load_calib(file):

	with open(file, "r") as f: 
		raw_calib = f.readlines()[:7]

	calib = {}
	rows = ['P0', 'P1', 'P2', 'P3', 'R0_rect', 'velo_to_cam', 'imu_to_velo']

	for i, (l, r) in enumerate(zip(raw_calib, rows)):
		l = l.split(' ')[1:]
		l[-1] = (l[-1][:-1])
		l = np.array(l).astype(np.float32)
		m = np.eye(4)
		l = l.reshape(3, -1)
		m[:l.shape[0], :l.shape[1]] = l
		calib[r] = m

	return calib


def load_label(file):

	with open(file, "r") as f: 
		raw_label = f.readlines()
	
	label = []
	for l in raw_label:
		l = l.split(' ')
		l[-1] = l[-1][:-1]
		l[0] = CLASS_MAP[l[0]]
		label.append(np.array(l).astype(np.float32))

	label = np.array(label)
	
	return label[label[:, 0]!=1]


def extract_2d_boxes(boxes_3d, calib, img_size):

	P_rect = calib['P2']

	boxes_3d = np.concatenate([boxes_3d, np.ones((*boxes_3d.shape[:2], 1))], -1)

	centers_2d = np.zeros((boxes_3d.shape[0], 2))
	boxes_2d = np.zeros((boxes_3d.shape[0], 4))

	for i, box in enumerate(boxes_3d):
		
		xyz_c = np.einsum('ij,kj->ki', P_rect, box)[:, :3]
		xy = xyz_c / xyz_c[:, 2, None]

		box_2d = np.array([
			np.min(xy[:, 0]), np.min(xy[:, 1]),
			np.max(xy[:, 0]), np.max(xy[:, 1]),
		])

		centers_2d[i] = [
			(box_2d[0] + ((box_2d[2] - box_2d[0]) / 2.)) / img_size[1],
			(box_2d[1] + ((box_2d[3] - box_2d[1]) / 2.)) / img_size[0],
		]
		
		boxes_2d[i] = np.array([
			np.clip(box_2d[0]/img_size[1], 0, 1), 
			np.clip(box_2d[1]/img_size[0], 0, 1),
			np.clip(box_2d[2]/img_size[1], 0, 1), 
			np.clip(box_2d[3]/img_size[0], 0, 1),
		])

	return centers_2d, boxes_2d

# TODO this function gives different values to 
# tf_utils.scenenet_to_nonaa()
def extract_3d_boxes(label):

	xyz = label[:, 11:14, None]
	hwl = label[:, 8:11, None]

	x_mins = xyz[:, 0] - (hwl[:, 2]/2.)
	x_maxs = xyz[:, 0] + (hwl[:, 2]/2.)

	y_mins = xyz[:, 1]
	y_maxs = xyz[:, 1] - hwl[:, 0]

	z_mins = xyz[:, 2] - (hwl[:, 1]/2.)
	z_maxs = xyz[:, 2] + (hwl[:, 1]/2.)

	boxes = np.concatenate(
		[x_mins, y_mins, z_mins, x_maxs, y_maxs, z_maxs]
	, 1)

	nonaa_boxes = np.zeros((boxes.shape[0], 8, 3))

	for i, b in enumerate(boxes):
		nonaa_boxes[i] = helpers.nonaa_box(b, label[i][14])

	return nonaa_boxes


def remove_dontcare(label):

	mask = np.squeeze(label['clf']==8)

	label['c_3d'][mask] = 0.
	label['c_2d'][mask] = 0.
	label['bbox_3d'][mask] = 0.
	label['bbox_2d'][mask] = 0.
	label['extent'][mask] = 0.
	label['ri'][mask] = 0.
	label['rj'][mask] = 0.

	return label


def scan_to_camera_coords(scan, calib):

	scan = np.hstack((scan, np.ones((scan.shape[0], 1))))

	v2c_mat = calib['velo_to_cam']
	cam_mat = calib['P2']
	r_mat = calib['R0_rect']

	R = calib['R0_rect'] @ calib['velo_to_cam']
		
	scan = np.einsum('ij,kj->ki', R, scan)[:, :3]

	return scan
