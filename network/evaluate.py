import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import toml
import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import MonoNet
from utils.dataset import load_dataset
from utils import tf_utils, helpers, eval, losses

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(42)


def evaluate():

    model = MonoNet(cfg['model'])
    model.load_weights(WEIGHTS)
    print('[info] model weights loaded.')

    dataset = load_dataset(DATASET, cfg, False)

    n_classes = cfg['model']['n_classes']

    cham_res = []
    ap = np.zeros((n_classes-1)) if n_classes > 1 else np.zeros((1))
    ap_count = np.zeros((n_classes-1)) if n_classes > 1 else np.zeros((1))

    for step, (inputs, label) in enumerate(dataset):

        calib = inputs['calib']

        pred_c, pred_attr, pred_clf = model([inputs['img'], calib['calib'], calib['img_orig']])

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

        label_mask = tf.cast(label['bbox_3d'], tf.bool)[:, :, 0, 0]
        label_clf = tf.boolean_mask(label['clf'], label_mask)

        label_c = tf.boolean_mask(label['c_3d'], label_mask)
        label_attr = tf.boolean_mask(label['attr'], label_mask)
        label_boxes = tf_utils.scenenet_to_aabb(
            tf.expand_dims(label_c, 0), tf.expand_dims(label_attr, 0)
        )

        cham_dist, _, _ = losses.chamfer_loss(label['c_3d'], tf.expand_dims(pred_c, 0))
        cham_res.append(cham_dist)

        res, classes = eval.ap_eval(
            tf.squeeze(label_boxes, 0),
            label_clf,
            pred_boxes,
            pred_clf,
            IOU_THRESH
        )

        for c_idx, c in enumerate(classes):
            ap[c] += res[c_idx][2]
            ap_count[c] += 1
        
        if step % 100 == 0 and step != 0:
            print('step: {}, Chamfer: {:.4f} AP: {:}, mAP: {:.2f}'.format(
                step, np.mean(cham_res), ap/ap_count, np.mean(ap/ap_count)))
            break

    print('-------------')
    print('step: {}, Chamfer: {:.4f} AP: {:}, mAP: {:.2f}'.format(
                step, np.mean(cham_res), ap/ap_count, np.mean(ap/ap_count)))

    return np.mean(ap/ap_count)


if __name__ == '__main__':

    LOG_DIR = './logs/kitti_car_1'
    MODEL_DIR = 'model'
    DATASET = './data/kitti_car_val.tfrecord'
    WEIGHTS = os.path.join(LOG_DIR, MODEL_DIR, 'weights.ckpt')
    SCORE_THRESH = 0.5
    NMS = True
    NMS_THRESH = 0.25
    IOU_THRESH = 0.5

    cfg = toml.load(os.path.join(LOG_DIR, 'config.toml'))
    cfg['model']['batch_size'] = 1

    score = evaluate()