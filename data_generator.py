import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,trains, labels, wv, batch_size=128, train_dim=(32,32,32), label_dim=(10,10), n_channels=128,
                 n_classes=None, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.trains= trains
        self.train_size=len(trains)
        self.labels = labels
        self.wv=wv
        self.train_dim=train_dim
        self.label_dim=label_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.train_size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = list(range(self.train_size))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization, the is used for classified model
#        X = np.empty((self.batch_size, *self.train_dim, self.n_channels))
#        y = np.empty((self.batch_size), dtype=int)

        # this is used for regression model
        encoder_input_data = np.zeros(
            (self.batch_size, self.train_dim, self.n_channels),
            dtype='float32')
        decoder_input_data = np.zeros(
            (self.batch_size, self.label_dim, self.n_channels),
            dtype='float32')
        decoder_target_data = np.zeros(
            (self.batch_size, self.label_dim, self.n_channels),
            dtype='float32')

        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample

            content=self.trains[ID]
            title=self.labels[ID]
            title = ['\t'] + title + ['\n']
            content_len=min(self.train_dim,len(content))
            title_len=len(title)
            content_vec = self.wv[content[0:content_len]]
            title_vec = self.wv[title]
            encoder_input_data[i, 0:content_len] = content_vec
            decoder_input_data[i,0:title_len] = title_vec
            decoder_target_data[i, 0:title_len - 1] = title_vec[1:title_len]
#            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
#            y[i] = self.labels[ID]
        X=[encoder_input_data,decoder_input_data]
        y=decoder_target_data

        # Return the data
        if self.n_classes:
            return X, y
        else:
            return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

    def info(self,list_id):
        return self.__data_generation(list_id)
