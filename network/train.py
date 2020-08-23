import os
import sys

sys.path.insert(0, './')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import toml
import tensorflow as tf
from tensorflow import keras

from utils import tf_utils
from model import MonoNet
from utils import helpers
from utils.dataset import load_dataset

tf.random.set_seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train():

    model = MonoNet(cfg['model'])
    
    if cfg['std']['pretrain']: 
        model.load_weights(cfg['std']['pretrain'])
        print('[info] pretrained weighted loaded from : {}'.format(cfg['std']['pretrain']))

    train_ds = load_dataset(cfg['std']['train_file'], cfg)
    val_ds = load_dataset(cfg['std']['val_file'], cfg)

    callbacks = [		
		keras.callbacks.TensorBoard(
			log_dir='./logs/{}'.format(cfg['std']['log_code']), update_freq='epoch', profile_batch='10,20'),
		keras.callbacks.ModelCheckpoint(
            './logs/{}/model/weights.ckpt'.format(cfg['std']['log_code']), save_weights_only=True, save_best_only=True)
	]

    helpers.dump_config(cfg)
    tf_utils.print_summary(model, cfg)
    
    model.compile(
        keras.optimizers.Adam(learning_rate=cfg['model']['lr']),
        keras.optimizers.Adam(learning_rate=cfg['model']['lr']),
        eager=True
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        validation_steps=1,
        validation_freq=1,
        callbacks=callbacks,
        epochs=cfg['model']['epochs'],
        steps_per_epoch=cfg['std']['val_freq'],
        verbose=0
    )

if __name__ == '__main__':

    try:
        cfg = toml.load(sys.argv[1])
    except:
        raise ValueError("No config file passed.")

    train()
