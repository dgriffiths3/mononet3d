import numpy as np
import tensorflow as tf

from utils import losses, helpers


def precision_recall(tp, fp, fn):

	if fn < 0: fn = 0

	precision = float(tp) / (tp + fp) if tp+fp > 0 else 0
	recall = float(tp) / (tp + fn)
	acc = tp/(tp+fp+fn)

	return [precision, recall, acc]


def calculate_ap(rec, prec):

	mrec = np.concatenate(([0.], rec, [1.]))
	mpre = np.concatenate(([0.], prec, [0.]))

	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

	i = np.where(mrec[1:] != mrec[:-1])[0]

	ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

	return ap


def ap_eval(label_boxes, label_clf, pred_boxes, pred_clf, iou_thresh=0.25):
	""" AABB IoU meanAP eval """

	classes, _ = tf.unique(tf.reshape(label_clf, -1))
	results = np.zeros((classes.shape[0], 3), np.float32)

	for c_idx, c in enumerate(classes):
		
		label_mask = tf.where(label_clf==c)[:, 0]
		pred_mask = tf.where(pred_clf==c)[:, 0]

		if pred_mask.shape[0] == 0:
			results += [0., 0., 0.]
			continue

		l_boxes = tf.gather(label_boxes, label_mask, axis=0)
		p_boxes = tf.gather(pred_boxes, pred_mask, axis=0)

		tp = np.zeros(p_boxes.shape[0], dtype=np.float32)
		fp = np.zeros(p_boxes.shape[0], dtype=np.float32)
		
		labels_used = []

		npos = l_boxes.shape[0]

		for p_idx, pred in enumerate(p_boxes):

			pos = False

			for l_idx, label in enumerate(l_boxes):

				if l_idx in labels_used: continue

				iou_score = helpers.iou(label, pred)

				if iou_score >= iou_thresh:
					labels_used.append(l_idx)
					pos = True
					tp[p_idx] = 1.
					break

			if pos == False: fp[p_idx] = 1.
		
		tp = np.cumsum(tp)
		fp = np.cumsum(fp)

		rec = tp / float(npos)
		prec = tp / np.maximum(tp + fp, 1e-10)

		ap = calculate_ap(rec, prec)

		m_prec = prec[-1]
		m_rec = rec[-1]

		results[c_idx] = [m_prec, m_rec, ap]

	return results, classes