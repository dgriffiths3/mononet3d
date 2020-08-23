import os

import numpy as np
import progressbar
import pyvista as pv
import toml
import tensorflow as tf
import colorsys


class EvalProgBar():

	def __init__(self):

		self.widgets = [progressbar.FormatLabel('')]
		self.bar = progressbar.ProgressBar(widgets=self.widgets)
		self.bar.start(max_value=progressbar.UnknownLength)

	def update(self, step, metrics):

		self.widgets[0] = progressbar.FormatLabel(
			'step: {}, AP: {:}, mAP: {:.2f}'.format(step, metrics[0], metrics[1])
		)
		self.bar.update(step, True)


def progbar(n):

	bar = progressbar.ProgressBar(
		maxval=n,
		widgets=[
			progressbar.Bar('=', '[', ']'), ' ',
			progressbar.Percentage(), ' | ',
			progressbar.SimpleProgress(), ' | ',
			progressbar.AdaptiveETA()
		]
	)

	return bar


def colormap(n_classes):

	vals = np.linspace(0, 1, n_classes)
	return np.array([colorsys.hsv_to_rgb(c, 0.8, 0.8) for c in vals])


def dump_config(cfg):

	save_dir = os.path.join('./logs/{}'.format(cfg['std']['log_code']))
	if not os.path.isdir(save_dir): os.makedirs(save_dir)
	f = open(os.path.join(save_dir, 'config.toml'), "w")
	s = toml.dumps(cfg)
	f.write(s)
	f.close()


def euc_dist(a, b):

	return np.sqrt(np.sum((a - b)**2))


def bounding_box(pc):

	bbox = [
		np.min(pc[:, 0]), np.min(pc[:, 1]), np.min(pc[:, 2]),
		np.max(pc[:, 0]), np.max(pc[:, 1]), np.max(pc[:, 2])
	]

	return np.array(bbox)


def bbox_overlap(pc_a, pc_b):

	bbox_a = bounding_box(pc_a)
	bbox_b = bounding_box(pc_b)

	if (
		bbox_a[3] >= bbox_b[0] and bbox_b[3] >= bbox_a[0] and
		bbox_a[4] >= bbox_b[1] and bbox_b[4] >= bbox_a[1] and
		bbox_a[5] >= bbox_b[2] and bbox_b[5] >= bbox_a[2]
		):
		overlap = True
	else:
		overlap = False

	return overlap


def nonaa_box(b, theta, axis=1):
	
	pts = np.array([
		[b[0], b[1], b[2]],
		[b[3], b[1], b[2]],
		[b[0], b[1], b[5]],
		[b[3], b[1], b[5]],
		[b[0], b[4], b[2]],
		[b[3], b[4], b[2]],
		[b[0], b[4], b[5]],
		[b[3], b[4], b[5]]
	])

	return rotate_euler(pts, theta, axis)


def make_lines(pts):

	lines = [
		pv.Line(pts[0], pts[1]), pv.Line(pts[1], pts[3]), pv.Line(pts[3], pts[2]), pv.Line(pts[2], pts[0]),
		pv.Line(pts[0], pts[4]), pv.Line(pts[1], pts[5]), pv.Line(pts[3], pts[7]), pv.Line(pts[2], pts[6]),
		pv.Line(pts[4], pts[6]), pv.Line(pts[6], pts[7]), pv.Line(pts[7], pts[5]), pv.Line(pts[5], pts[4])
	]

	return lines


def rotate_euler(pts, theta, axis=2, center=None):

	c = np.cos(theta)
	s = np.sin(theta)

	if axis == 0:
		R = np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])
	elif axis == 1:
		R = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
	elif axis == 2:
		R = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

	mean = np.mean(pts, axis=0) if center is None else center

	pts -= mean
	pts = np.einsum('ij,kj->ki', R, pts)
	pts += mean

	return pts


def get_fixed_pts(in_pts, n_pts):

	out_pts = np.zeros((n_pts, 3))
	ret = True

	if in_pts.shape[0] == 0:
		ret = False
	elif in_pts.shape[0] < n_pts:
		out_pts[0:in_pts.shape[0]] = in_pts
		s_idx = np.arange(n_pts)
		np.random.shuffle(s_idx)
		out_pts = out_pts[s_idx]
	else:
		s_idx = np.arange(in_pts.shape[0])
		np.random.shuffle(s_idx)
		out_pts = in_pts[s_idx[0:n_pts]]

	return ret, out_pts


def iou(a, b):

	xx1 = np.maximum(a[0], b[0])
	yy1 = np.maximum(a[1], b[1])
	zz1 = np.maximum(a[2], b[2])

	xx2 = np.minimum(a[3], b[3])
	yy2 = np.minimum(a[4], b[4])
	zz2 = np.minimum(a[5], b[5])

	w = np.maximum(0.0, xx2 - xx1)
	h = np.maximum(0.0, yy2 - yy1)
	d = np.maximum(0.0, zz2 - zz1)

	inter = w * h * d

	area_a = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
	area_b = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])

	return inter / float(area_a + area_b - inter)


def nms(boxes, scores, max_out=100, iou_thresh=0.25):
	"""
	Code adapted from : https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
	"""

	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	z1 = boxes[:, 2]

	x2 = boxes[:, 3]
	y2 = boxes[:, 4]
	z2 = boxes[:, 5]

	keep_inds = []

	if scores.shape[0] > 0:

		order = np.argsort(-scores, axis=0)

		areas = (x2 - x1) * (y2 - y1) * (z2 - z1)

		num_in = 0

		while order.size > 0:

			if num_in == max_out: break

			i = order[0]

			keep_inds.append(i[0])

			num_in += 1

			xx1 = np.maximum(x1[i], x1[order[1:]])
			yy1 = np.maximum(y1[i], y1[order[1:]])
			zz1 = np.maximum(z1[i], z1[order[1:]])

			xx2 = np.minimum(x2[i], x2[order[1:]])
			yy2 = np.minimum(y2[i], y2[order[1:]])
			zz2 = np.minimum(z2[i], z2[order[1:]])

			w = np.maximum(0.0, xx2 - xx1)
			h = np.maximum(0.0, yy2 - yy1)
			d = np.maximum(0.0, zz2 - zz1)

			inter = w * h * d
			ovr = inter / (areas[i] + areas[order[1:]] - inter)

			inds = np.where(ovr <= iou_thresh)[0]
			order = order[inds + 1]

	return keep_inds


def kitti_scenenet_to_oabb(center, attr):
	
	extent = attr[:, :3]
	
	theta = np.arctan2(attr[:, 4], attr[:, 3])

	x_mins = center[:, 0, None] - (extent[:, 2, None]/2.)
	x_maxs = center[:, 0, None] + (extent[:, 2, None]/2.)
	
	y_mins = center[:, 1, None]
	y_maxs = center[:, 1, None] - (extent[:, 0, None])
	
	z_mins = center[:, 2, None] - (extent[:, 1, None]/2.)
	z_maxs = center[:, 2, None] + (extent[:, 1, None]/2.)

	boxes = np.concatenate(
		[x_mins, y_mins, z_mins, x_maxs, y_maxs, z_maxs]
	, 1)

	nonaa_boxes = np.zeros((boxes.shape[0], 8, 3))

	for i, b in enumerate(boxes):
		nonaa_boxes[i] = nonaa_box(b, theta[i])

	return nonaa_boxes


def get_fixed_pts(in_pts, n_pts):

	out_pts = np.zeros((n_pts, 3))
	ret = True

	if in_pts.shape[0] == 0:
		ret = False
	elif in_pts.shape[0] < n_pts:
		out_pts[0:in_pts.shape[0]] = in_pts
		s_idx = np.arange(n_pts)
		np.random.shuffle(s_idx)
		out_pts = out_pts[s_idx]
	else:
		s_idx = np.arange(in_pts.shape[0])
		np.random.shuffle(s_idx)
		out_pts = in_pts[s_idx[0:n_pts]]

	return ret, out_pts
