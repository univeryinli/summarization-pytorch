from keras import optimizers, losses, metrics
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing import sequence
from keras.utils import multi_gpu_model
import pickle,os
from compiler.ast import flatten
#from parallel_model import ParallelModel
from loss_history import LossHistory
from multiprocessing import Pool
data_dim = 128
import gc,random
import numpy as np


def batch_generator(data,batch_size,shuffle=True):
    count = 0
    if shuffle:
        print('shuffle start!')
        shuffle_index=list(range(data[0].shape[0]))
        random.shuffle(shuffle_index)
        data = [b_data[shuffle_index] for b_data in data]
        print('shuffle is done!')
    while True:
        start = count * batch_size
        end = (count + 1) * batch_size
        count += 1
        if (count + 1) * batch_size > data[0].shape[0]:
            count = 0
        train=sequence.pad_sequences(data[0][start:end], maxlen=2190, padding='post')
        label=data[1][start:end]
        yield [train,label]


def cos(y_true, y_pre):
    return losses.cosine(y_true, y_pre)


def get_model(timesteps, data_dim,output_dim):
    inputs = Input(shape=(timesteps,data_dim))
    print('inputs.shape:', inputs.shape)
    lstm1_1 = LSTM(128, return_sequences=True)(inputs)
    lstm1_2 = LSTM(256, return_sequences=True)(lstm1_1)
    lstm1_3 = LSTM(256)(lstm1_2)

    lstm2_1 = LSTM(128, return_sequences=True)(inputs)
    lstm2_2 = LSTM(256, return_sequences=True)(lstm2_1)
    lstm2_3 = LSTM(256)(lstm2_2)

    lstm3_1 = LSTM(256, return_sequences=True)(inputs)
    lstm3_2 = LSTM(256, return_sequences=True)(lstm3_1)
    lstm3_3 = LSTM(256)(lstm3_2)


    dense0 = concatenate([lstm1_3, lstm2_3, lstm3_3])
    dense1 = Dense(768, activation='relu')(dense0)
    dense2 = Dense(1024, activation='relu')(dense1)
    dense3 = Dense(2048, activation='relu')(dense2)
    outputs = Dense(output_dim, activation='relu')(dense3)
    print('output.shape',outputs.shape)

    model1 = Model(inputs=inputs, outputs=outputs)
    sgd = optimizers.SGD()
    model1 = multi_gpu_model(model1, 2)
    model1.compile(optimizer=sgd, loss='cosine', metrics=['mse'])
    return model1

'''
def list_read(filename,flag=False):
    # Try to read a txt file and return a list.Return [] if there was a
    # mistake.
    try:
        file1 = open(filename, 'r')
    except IOError:
        error = []
        return error
    print('listread:' + filename + 'start!')
    lines=[]
    for line in file1:
        if flag:
            if type(line).__name__ == 'str':
                line = eval(line).flatten()
                lines.append(line)
            elif type(line).__name__ == 'list':
                line =line.flatten()
                lines.append(line)
#            vector=wv[line]
#            file2.write(str(vector))
        else:
            line = eval(line)
            lines.append(line)
    return lines
    print('listread:' + filename + 'done!')
'''
''''
def list_read(filename,flag=False):
    # Try to read a txt file and return a list.Return [] if there was a
    # mistake.
    try:
        file1 = open(filename, 'r')
    except IOError:
        error = []
        return error
    print('listread:' + filename + 'start!')
    lines=[]
    for line in file1:
        if flag:
            line=np.fromstring(line,dtype=np.float32)
            line=line.flatten()
            lines.append(line)
        else:
            line = np.fromstring(line, dtype=np.float32)
            lines.append(line)
    return lines
    print('listread:' + filename + 'done!')
'''

def list_read(filename,flag=False):
    print('listread:' + filename + 'start!')
    vectors=np.load(filename)
    return vectors



def multu_process_file(path,IsReturn=False,IsFlatten=False):

    print('Parent process %s.', os.getpid())
    listdir=os.listdir(path)
    workers=5
    print('workers:',workers)
    p = Pool(workers)
    res_list=[]
    for i in range(workers):
        for dir in listdir:
            if ('npy' in dir) and (str(i).zfill(2) in dir):
                pathread=path+dir
                print(pathread)
                res_list.append(p.apply_async(list_read, (pathread,)))
        print('task:',i)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    if IsReturn:
        results = []
        for i in range(len(res_list)):
            results=results+list(res_list[i].get())
            res_list[i]=0
        del res_list
        gc.collect()
        return results
    print('All subprocesses done.')


path_base = '../data/'
path_train = path_base + 'train_2/'
path_label = path_base + 'label_2/'
path_xdata=path_base+'xdata/'
path_ydata=path_base+'ydata/'
#titles=multu_process_file(path_label,IsReturn=True)
titles=np.load(path_label+'label_pad.npy')
#titles = sequence.pad_sequences(titles,dtype='float32',padding='post')
#np.save(path_label+'label_pad',titles)
y_data = titles
del titles
gc.collect()
print('y_data is done!')

contents=np.load(path_train+'train.npy')
contents=np.array(contents)
print(type(contents),type(contents[0]))
#contents=multu_process_file(path_train,IsReturn=True)
#contents = sequence.pad_sequences(contents,dtype='float32', padding='post')
#np.save(path_train+'train',contents)
print('x_data is done!')
#timesteps = len(contents[0])
timesteps=2190
x_data = contents
del contents
gc.collect()

data_size=len(x_data)
train_size = int(data_size*0.8)
x_train, y_train = x_data[0:train_size], y_data[0:train_size]
output_dim=len(y_train[0])
print('x_train_size:', len(x_train))
print('y_train_size:', len(y_train))
x_test,y_test=x_data[train_size-1:data_size-1],y_data[train_size-1:data_size-1]
print('x_test_size:', len(x_test))
print('y_test_size:', len(y_test))
print('start lstm')

train_generator=batch_generator([x_train,y_train],100)
test_generator=batch_generator([x_test,y_test],100)

model = get_model(timesteps, data_dim,output_dim)
model.summary()
earlystop = EarlyStopping()
losshistory= LossHistory()
modelcheck=ModelCheckpoint(path_base+'bestmodel.m',monitor='val_loss',verbose=1,save_best_only=True)
model.fit_generator(train_generator,steps_per_epoch=100,epochs=10,callbacks=[losshistory,modelcheck],validation_data=test_generator,validation_steps=5)
losshistory.save('./history.json')
losshistory.loss_plot('epoch')
losshistory.loss_plot('batch')
