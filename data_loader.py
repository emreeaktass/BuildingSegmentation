import dataset as D
from tensorflow.keras.utils import Sequence
import random
import numpy as np
import matplotlib.pyplot as plt



class DataLoader(Sequence):
    def __len__(self):

        return int(np.ceil(len(self.X) / self.batch_size))

    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y
        self.indices = list(range(len(self.X)))
        self.length = len(self.X)

        self.batch_size = batch_size
        random.shuffle(self.indices)

        self.xrange = 0
        if self.length % batch_size == 0:
            self.xrange = self.length // batch_size
        else:
            self.xrange = self.length // batch_size + 1

        self.batches_indices = [ self.indices[a:]
             if (self.length - a < self.batch_size) else self.indices[a:a + self.batch_size] for a in
            range(0, self.length, self.batch_size)]



    def __getitem__(self, idx):

        self.batch_train = []
        self.batch_test = []

        for i in range(len(self.batches_indices[idx])):
            self.batch_train.append(np.load(self.X[self.batches_indices[idx][i]]))
            self.batch_test.append(np.load(self.y[self.batches_indices[idx][i]]))
        self.data = [i for i in self.batch_train]
        self.mask = [i for i in self.batch_test]

        # print(self.data[0].shape)
        self.data = np.array(self.data).astype(np.float32)
        # print(self.data.shape)
        self.mask = np.array(self.mask).astype(np.float32)


        self.data = self.data / 255.
        self.data = self.data.reshape((len(self.data), 256, 256, 3))
        self.mask = self.mask.reshape((len(self.mask), 256, 256))
        # print(self.mask.shape)
        return self.data, self.mask


def get_generator(X_train, y_train, batch_size):
    return DataLoader(X_train, y_train, batch_size)


if __name__ == '__main__':
    datas, masks = D.get_splitted_datas('train')
    dl = DataLoader(datas, masks, 8)

    for i in dl:
        print(i[0].shape, i[1].shape)
        print(i[0][1].shape)
        print(i[1][1].shape)
        a = i[0][1]
        b = i[1][1]
        d = np.hstack((a[:, :, 0], b))
        # plt.figure(1)
        # plt.imshow(data[:,:,0], cmap='gray')
        # plt.figure(2)
        # plt.imshow(mask, cmap='gray')
        plt.imshow(d, cmap='gray')
        plt.show()