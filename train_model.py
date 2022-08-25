import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
import random
import numpy as np
import argparse
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import BaseLogger
from tensorflow.keras.utils import to_categorical
from model import cnn_model, lstm_model

def parse_args():
    parser = argparse.ArgumentParser(description='Training Teacher self-supervised learning')
    parser.add_argument('--model', type=str, default='cnn', help='cnn or lstm')
    parser.add_argument('--gpu', type=str, default='1', help='GPU id')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='drop rate')
    parser.add_argument('--reg_factor', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--start_epoch', type=int, default=1, help='manual epoch number (useful on restarts)')

    args = parser.parse_args()
    return args

def load_data_npz(npz_path):
    d = np.load(npz_path)
    return d["X"], d["y"]

class SaveLog(BaseLogger):
    def __init__(self, jsonPath=None, jsonName=None, startAt=0, verbose=0):
        super(SaveLog, self).__init__()
        self.jsonPath = jsonPath
        self.jsonName = jsonName
        self.jsonfile = os.path.join(self.jsonPath, self.jsonName)
        self.startAt = startAt
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the JSON history path exists, load the training history
        if self.jsonfile is not None:
            if os.path.exists(self.jsonfile):
                self.H = json.loads(open(self.jsonfile).read())

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc.
        # for the entire training process

        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(float(v))
            self.H[k] = l

        # check to see if the training history should be serialized
        # to file
        if self.jsonfile is not None:
            f = open(self.jsonfile, "w")
            f.write(json.dumps(self.H))
            f.close()


def build_callbacks(save_path, model_type='cnn', monitor='val_loss', mode='min', startAt=1):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    checkpoint_path = os.path.join(save_path, 'checkpoints')
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    jsonPath = os.path.join(save_path, "output")
    if not os.path.exists(jsonPath):
        os.mkdir(jsonPath)

    earlyStopping = EarlyStopping(monitor=monitor, patience=20, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=5, verbose=1, factor=0.1, min_lr=1e-8)

    # Save log file during training
    jsonName = model_type + '.json'
    saveLog = SaveLog(jsonPath=jsonPath, jsonName=jsonName, startAt=startAt, verbose=1)

    # Save the best model to .h5 files
    filepath = os.path.join(save_path, model_type + '.h5')
    model_checkpoint = ModelCheckpoint(filepath, verbose=1, monitor='val_accuracy', mode='max', save_best_only=True,
                                       save_weights_only=True)
    cb = [model_checkpoint, earlyStopping, reduce_lr, saveLog]
    return cb

def train(X_train, y_train, X_test, y_test, batch_size=64, model_type='cnn', epochs=200, lr_init=0.001, start_epoch=1, drop_rate=0.5,
          save_path='./'):
    # Build model
    if model_type == 'cnn':
        model = cnn_model(drop_rate=drop_rate)
    else:
        model = lstm_model(drop_rate=drop_rate)
    model.summary(line_length=150)

    # Optimization and Loss
    sgd = SGD(learning_rate=lr_init, momentum=0.9, nesterov=True)

    # Build call backs
    cb = build_callbacks(save_path, model_type=model_type, monitor='val_loss', mode='min', startAt=start_epoch)

    loss = CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer=sgd, loss=loss, metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=epochs - start_epoch + 1, verbose=1, batch_size=batch_size,
              validation_data=(X_test, y_test), callbacks=cb)

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # Choose GPU for training

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    model_type = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    lr_init = args.lr
    start_epoch = args.start_epoch
    drop_rate = args.drop_rate

    npz_path = 'data.npz'
    X, y = load_data_npz(npz_path=npz_path)
    y = y-1
    y = to_categorical(y, num_classes=3)

    # scaler = MinMaxScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    print('Train X shape:', X_train.shape)
    print('Train y shape:', y_train.shape)
    print('Test X set:', X_test.shape)
    print('Test y shape:', y_test.shape)

    save_path = 'save_models'
    train(X_train, y_train, X_test, y_test, batch_size=batch_size, model_type=model_type, epochs=epochs,
          lr_init=lr_init, start_epoch=start_epoch, drop_rate=drop_rate, save_path=save_path)


if __name__ == '__main__':
    print(tf.__version__)
    main()