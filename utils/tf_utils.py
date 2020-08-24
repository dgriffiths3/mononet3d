import sys
sys.path.insert(0, './')

import tensorflow as tf
from utils import helpers


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def print_summary(model, cfg):

	img_size = cfg['model']['img_size']
	input_shape = [
		(cfg['model']['batch_size'], *img_size),
		(cfg['model']['batch_size'], 4, 4),
		(cfg['model']['batch_size'], 2)
	]

	model.build(input_shape)

	shape = input_shape[0] 
	lr = cfg['model']['lr'] 
	log_code = cfg['std']['log_code']

	print(model.summary())
	print('------------------------------------------')
	print('Input shape: {}'.format(shape))
	print('Number epochs: {}'.format(cfg['model']['epochs']))
	print('Batch size: {}'.format(cfg['model']['batch_size']))
	print('Learning rate: {}'.format(lr))
	print('Log name: {}'.format(log_code))
	print('------------------------------------------')


def pts_to_boxes(pts, offset):

	boxes = tf.reshape(pts, (-1, 3))
	min = tf.subtract(boxes, offset/2.)
	max = tf.add(boxes, offset/2.)
	boxes = tf.reshape(
		tf.stack([min, max], axis=1),
		(pts.shape[0], pts.shape[1], 6)
	)

	return boxes


def masked_conf(prob, n_classes):

	conf_max = tf.reduce_max(prob, -1)
	conf_class = tf.cast(tf.math.argmax(prob, -1), tf.int32)
	conf = tf.where(conf_class != n_classes-1, conf_max, tf.zeros_like(conf_max))

	return conf


def objectness_mask(center, attr, clf, scores, thresh=0.9):
	""" Will only work when batch size = 1 """

	assert tf.rank(center) == 2, 'Objectness must be single batch'

	mask = tf.where(scores>=thresh)[:, 0]

	center = tf.gather(center, mask, axis=0)
	attr = tf.gather(attr , mask, axis=0)
	clf = tf.gather(clf, mask, axis=0)
	scores = tf.gather(scores, mask, axis=0)

	if tf.rank(center) == 1:
		center = tf.expand_dims(center, 0)
		attr = tf.expand_dims(attr, 0)
		clf = tf.expand_dims(clf, 0)
		scores = tf.expand_dims(scores, 0)

	return center, attr, clf, scores


def obj_to_cam_crop(pts, images, crop_size, calib, o_img_size):

	img_size = tf.cast(images.shape[1:3], tf.float32)

	pts = tf.concat([pts, tf.ones([*pts.shape[:2], 1])], -1)

	xyz_c = tf.einsum('...ij,...kj->...ki', calib, pts)[:, :, :3]
	uv = (xyz_c[:, :, :3] / xyz_c[:, :, 2, None])[:, :, :2]

	uv = tf.reverse(uv, [-1]) / o_img_size[:, None, :]

	hw = tf.cast(crop_size[:2], tf.float32) / 2. / img_size
	
	boxes = tf.concat([uv-hw, uv+hw], -1)
	boxes = tf.reshape(boxes, (boxes.shape[0]*boxes.shape[1], 4))

	inds = tf.sort(tf.tile(tf.range(images.shape[0]), [pts.shape[1]]))

	return  tf.image.crop_and_resize(images, boxes, inds, [*crop_size[:2]])


def oaaa_to_aabb(pts):

	aabb = tf.stack([
		tf.reduce_min(pts[:, :, :, 0], 2),
		tf.reduce_min(pts[:, :, :, 1], 2),
		tf.reduce_min(pts[:, :, :, 2], 2),
		tf.reduce_max(pts[:, :, :, 0], 2),
		tf.reduce_max(pts[:, :, :, 1], 2),
		tf.reduce_max(pts[:, :, :, 2], 2)
	], 2)

	return aabb


def scenenet_to_aabb(center, attr):

	extent = attr[:, :, :3]

	theta = tf.math.atan2(attr[:, :, 4], attr[:, :, 3])

	sinval = tf.math.sin(theta)
	cosval = tf.math.cos(theta)

	mins = center - (extent / 2.)
	maxs = center + (extent / 2.)
	
	box_pts = tf.stack([
		tf.stack([mins[:, :, 0], mins[:, :, 1], mins[:, :, 2]], -1),
		tf.stack([mins[:, :, 0], maxs[:, :, 1], mins[:, :, 2]], -1),
		tf.stack([maxs[:, :, 0], mins[:, :, 1], mins[:, :, 2]], -1),
		tf.stack([maxs[:, :, 0], maxs[:, :, 1], mins[:, :, 2]], -1),
		tf.stack([mins[:, :, 0], mins[:, :, 1], maxs[:, :, 2]], -1),
		tf.stack([mins[:, :, 0], maxs[:, :, 1], maxs[:, :, 2]], -1),
		tf.stack([maxs[:, :, 0], mins[:, :, 1], maxs[:, :, 2]], -1),
		tf.stack([maxs[:, :, 0], maxs[:, :, 1], maxs[:, :, 2]], -1),
	], 2)

	n_R = center.shape[0] * center.shape[1]

	c = tf.reshape(tf.math.cos(theta), (-1,))
	s = tf.reshape(tf.math.sin(theta), (-1,))
	z = tf.zeros((n_R,), tf.float32)
	o = tf.ones((n_R,), tf.float32)

	R = tf.reshape(
		tf.stack([c, z, s, z, o, z, -s, z, c], -1), 
		(*center.shape[:2], 3, 3)
	)

	box_means = tf.reduce_mean(box_pts, axis=2, keepdims=True)
	box_pts -= box_means
	
	box_pts_r = tf.einsum('...ij,...kj->...ki', R, box_pts)

	box_pts_r += box_means

	return oaaa_to_aabb(box_pts_r)


def nn_distance(a, b):

	n = a.shape[1]
	m = b.shape[1]

	dist_mask = tf.cast(a, tf.bool)[:, :, None, 0]
	mask_inds = tf.reshape(tf.tile(dist_mask, [1, 1, m]), (a.shape[0], n, m))
	mask_inds = tf.where(tf.math.logical_not(mask_inds))
	fill = tf.zeros_like(mask_inds, tf.float32)[:, 0] + 1e10

	a_tile = tf.expand_dims(a, 2)
	b_tile = tf.expand_dims(b, 1)

	a_tile = tf.tile(a_tile, [1, 1, m, 1])
	b_tile = tf.tile(b_tile, [1, n, 1, 1])

	diff = tf.reduce_sum(tf.square(a_tile-b_tile), -1)
	diff = tf.tensor_scatter_nd_update(diff, mask_inds, fill)

	dist1 = tf.reduce_min(diff, 2)[:, :, None]
	dist1 = tf.boolean_mask(dist1, tf.squeeze(dist_mask, -1))
		
	idx1 = tf.argmin(diff, 2)[:, :, None]

	dist2 = tf.reduce_min(diff, 1)[:, :, None]
	idx2 = tf.argmin(diff, 1)[:, :, None]
	
	return dist1, idx1, dist2, idx2