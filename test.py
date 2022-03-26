import tensorflow as tf
from sklearn.model_selection import train_test_split
import dataset as D
import data_loader as DL
import model as M
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
    X_test, y_test = D.get_splitted_datas('test')
    test_generator = DL.get_generator(X_test, y_test, 1)

    X = []
    y = []

    for i in test_generator:
        X.append(i[0][0, :, :, :])
        y.append(i[1][0, :, :])
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)


    model = tf.keras.models.load_model('trained_model_v5/', compile =False)
    y_pred = model.predict(X)
    import matplotlib.pyplot as plt

    row = 2
    col = 2
    fig1 = plt.figure(1, figsize=(200, 200))

    for i in range(1, col * row + 1):
        fig1.add_subplot(row, col, i)
        fig1.set_size_inches(18.5, 10.5, forward=True)
        xx = y_pred[i]
        print('xx : ', xx.max(), y[i].max())
        xx[xx >= 0.5] = 1
        xx[xx < 0.5] = 0
        xx = xx.reshape((256, 256))
        res = np.hstack((xx, X[i,:,:,0], y[i]))
        plt.imshow(res, cmap='gray')

plt.show()

