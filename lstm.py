from keras import optimizers, losses, metrics
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.utils import multi_gpu_model
import pickle
#from parallel_model import ParallelModel
from lossHistory import LossHistory

data_dim = 128


def cos(y_true, y_pre):
    return losses.cosine(y_true, y_pre)


def get_model(timesteps, data_dim):
    inputs = Input(shape=(timesteps, data_dim))
    print('inputs.shape:', inputs.shape)
    lstm1_1 = LSTM(128, input_shape=(timesteps, data_dim), return_sequences=True)(inputs)
    lstm1_2 = LSTM(256, return_sequences=True)(lstm1_1)
    lstm1_3 = LSTM(256)(lstm1_2)

    conv1 = Conv1D(128, 5, input_shape=(timesteps, data_dim), padding='same', activation='relu')(inputs)
    lstm2_1 = LSTM(128, return_sequences=True)(conv1)
    lstm2_2 = LSTM(256, return_sequences=True)(lstm2_1)
    lstm2_3 = LSTM(256)(lstm2_2)

    conv2 = Conv1D(256, 5, input_shape=(timesteps,data_dim), padding='same', activation='relu')(inputs)
    lstm3_1 = LSTM(256, return_sequences=True)(conv2)
    lstm3_2 = LSTM(256, return_sequences=True)(lstm3_1)
    lstm3_3 = LSTM(256)(lstm3_2)

    dense0 = concatenate([lstm1_3, lstm2_3, lstm3_3])
    dense1 = Dense(768, activation='relu')(dense0)
    dense2 = Dense(1024, activation='relu')(dense1)
    dense3 = Dense(2048, activation='relu')(dense2)
    outputs = Dense(2816, activation='relu')(dense3)
    print('output.shape',outputs.shape)

    model1 = Model(inputs=inputs, outputs=outputs)
    sgd = optimizers.SGD()
    model1 = multi_gpu_model(model1, 2)
    model1.compile(optimizer=sgd, loss='mse', metrics=['acc'])
    return model1


path_base = '../data_sample/'
file1 = open(path_base + 'vectors_content.pickle', 'rb')
contents = pickle.load(file1)
titles = pickle.load(file1)
file1.close()

titles = [title.flatten() for title in titles]
titles = sequence.pad_sequences(titles,dtype='float32',padding='post')
y_data = titles

contents = sequence.pad_sequences(contents,dtype='float32', padding='post')

timesteps = len(contents[0])

'''
#contents=list(contents)
#temp=[]
#for content in contents:
#    content=list(content)
#    temp=temp+content
#contents=temp
'''
x_data = contents
train_size = len(x_data)
x_train, y_train = x_data[0:train_size], y_data[0:train_size]
print('x_train_size:', len(x_train))
print('y_train_size:', len(y_train))
# x_test,y_test=x_data[train_size:train_size+2],y_data[train_size:train_size+2]
print('start lstm')
model = get_model(timesteps, data_dim)

model.summary()
earlystop = EarlyStopping()
losshistory= LossHistory()
model.fit(x=x_train, y=y_train,batch_size=40, epochs=8, callbacks=[losshistory, earlystop], validation_split=0.2)
losshistory.save('./history.json')
losshistory.loss_plot('epoch')
losshistory.loss_plot('batch')
