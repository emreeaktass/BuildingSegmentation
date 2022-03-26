import tensorflow as tf

from sklearn.model_selection import train_test_split
import dataset as D
import data_loader as DL
import mapnet as M
import os
import numpy as np
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    print('Training Started...\n')
    X_train, y_train = D.get_splitted_datas('train')
    X_val, y_val = D.get_splitted_datas('validation')
    X_test, y_test = D.get_splitted_datas('test')
    train_generator = DL.get_generator(X_train, y_train, 4)
    val_generator = DL.get_generator(X_val, y_val, 4)
    test_generator = DL.get_generator(X_test, y_test, 1)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=1,
        mode='auto', baseline=None, restore_best_weights=True
    )

    model = M.mapnet()
    model.fit(train_generator, validation_data=val_generator, epochs=100, verbose=1, shuffle=True,
              callbacks=[early_stop])
    model.save('trained_model_v5/')

    print('Training Completed...\n')