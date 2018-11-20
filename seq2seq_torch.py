# -*- coding: utf-8 -*-
import torch, copy, time,sys,json,random,os
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from iteration_utilities import flatten
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True
python_version=int(sys.version[0])
batch_size = 128
vec_dim=256
max_encoder_seq_length = 100
max_decoder_seq_length=5
path_base = '../data/'


class Encoder1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder1, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embed_dropout=nn.Dropout()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=1,batch_first=True)
        self.drop_out=nn.Dropout()

    def forward(self, encode_inputs):
        #seq_length=encode_inputs.size()[1]
        self.rnn.flatten_parameters()
        encode_inputs=self.embed_dropout(encode_inputs)
        encode_outputs, encode_hidden = self.rnn(encode_inputs)
        #encode_outputs=self.drop_out(encode_outputs)
        #encode_outputs = encode_outputs.view( -1,seq_length ,vec_dim)
        #encode_hidden=encode_hidden.view(-1,1,self.hidden_size)
        encode_hidden=encode_hidden[0]
        return encode_outputs, encode_hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, batch_size/2, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            return result
        else:
            return result


class Encoder2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder2, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embed_dropout=nn.Dropout()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.lstm3 = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=1,batch_first=True)
        self.drop_out=nn.Dropout()

    def forward(self, encode_inputs):
        #seq_length=encode_inputs.size()[1]

        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        #encode_inputs=self.embed_dropout(encode_inputs)
        encode_outputs, encode_hidden = self.lstm1(encode_inputs)
        encode_outputs,encode_hidden=self.lstm2(encode_outputs,encode_hidden)
        encode_outputs,encode_hidden=self.lstm3(encode_outputs,encode_hidden)
        #encode_outputs=self.drop_out(encode_outputs)
        #encode_outputs = encode_outputs.view( -1,seq_length ,vec_dim)
        #encode_hidden=encode_hidden.view(-1,1,self.hidden_size)
        encode_hidden=(encode_hidden[0][0],encode_hidden[1][0])
        return encode_outputs, encode_hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, batch_size/2, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            return result
        else:
            return result


class Encoder3(nn.Module):
    def __init__(self, voca_size,embedd_size, hidden_size):
        super(Encoder3, self).__init__()
        self.hidden_size = hidden_size
        self.voca_size = voca_size
        self.embed_size=embedd_size

        self.embed=nn.Embedding(self.voca_size,embedd_size)
        self.lstm1 = nn.LSTM(input_size=embedd_size, hidden_size=hidden_size, num_layers=1,batch_first=True,bidirectional=True)

    def forward(self, encode_inputs, decode_inputs):
        #seq_length=encode_inputs.size()[1]

        self.lstm1.flatten_parameters()
        #encode_inputs=self.embed_dropout(encode_inputs)
        encode_inputs = self.embed(encode_inputs)
        decode_inputs = self.embed(decode_inputs)
        encode_outputs, encode_hidden = self.lstm1(encode_inputs)
        #encode_outputs=self.drop_out(encode_outputs)
        encode_hidden=tuple(i[0] for i in encode_hidden)
        return encode_outputs, encode_hidden, decode_inputs

    def init_hidden(self):
        result = Variable(torch.zeros(1, batch_size/2, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            return result
        else:
            return result


class AttnDecoder1(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(AttnDecoder1, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.embed_dropout = nn.Dropout()
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.rnn = nn.RNN(self.hidden_size, self.hidden_size, 1,batch_first=True)
        self.out = nn.Linear(self.hidden_size,self.hidden_size)
        self.dropout = nn.Dropout()

    def forward(self, decode_input, decode_hidden, encode_outputs):
        self.rnn.flatten_parameters()
        #decode_input=self.embed_dropout(decode_input)

        decode_batch_size, decode_length = decode_input.size()[0], decode_input.size()[1]
        decode_outputs=decode_input.new_zeros(decode_input.size())

        for i in range(decode_batch_size):
            batch_hidden = decode_hidden[i].unsqueeze(0).unsqueeze(0)
            batch_encode_output = encode_outputs[i]
            for j in range(decode_length):
                input_tensor = decode_input[i,j].view(1, -1)
                is_stop = torch.mean(input_tensor)
                if is_stop == 0:
                    break

                attn_weights = F.softmax(self.attn(torch.cat((input_tensor, batch_hidden[0]), 1)), dim=0)

                attn_applied = torch.bmm(attn_weights.unsqueeze(0), batch_encode_output.unsqueeze(0))
                decode_output = torch.cat((input_tensor, attn_applied[0]), 1)
                decode_output = self.attn_combine(decode_output).unsqueeze(0)

                decode_output = F.relu(decode_output)

                decode_output, batch_hidden = self.rnn(decode_output, batch_hidden)
                decode_output=self.out(decode_output)
                decode_outputs[i, j] = decode_output.view(-1)
        #decode_outputs=self.dropout(decode_outputs)
        return decode_outputs

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            return result
        else:
            return result


class AttnDecoder2(nn.Module):
    def __init__(self, hidden_size, max_length):
        super(AttnDecoder2, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.teacher_forcing_ratio=0.5

        self.embed_dropout = nn.Dropout()
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.lstm1 = nn.LSTM(self.hidden_size, self.hidden_size, 1,batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size, self.hidden_size ,1,batch_first=True)
        self.lstm3 = nn.LSTM(self.hidden_size, self.hidden_size, 1,batch_first=True)
        self.out = nn.Linear(self.hidden_size,self.hidden_size)
        self.dropout = nn.Dropout()

    def forward(self, decode_input, decode_hidden, encode_outputs):
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        self.lstm3.flatten_parameters()
        #decode_input=self.embed_dropout(decode_input)
        decode_batch_size, decode_length = decode_input.size()[0], decode_input.size()[1]
        decode_outputs = decode_input.new_zeros(decode_input.size())
        attn = decode_input.new_zeros(decode_batch_size,decode_length,self.max_length)

        for i in range(decode_batch_size):
            batch_hidden = decode_hidden[0][i].unsqueeze(0).unsqueeze(0)
            batch_cell=decode_hidden[1][i].unsqueeze(0).unsqueeze(0)
            batch_encode_output = encode_outputs[i]
            use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio and self.training==True else False
            input_tensor = decode_input[i,0].view(1, -1)

            for j in range(decode_length):
                if use_teacher_forcing:
                    input_tensor = decode_input[i,j].view(1, -1)
                #is_stop = torch.mean(input_tensor)
                #if is_stop == 0:
                #    break

                attn_weights = F.softmax(self.attn(torch.cat((input_tensor, batch_hidden[0]), 1)), dim=1)
                attn[i,j]=attn_weights[0]
                attn_applied = torch.bmm(attn_weights.unsqueeze(0), batch_encode_output.unsqueeze(0))
                decode_output = torch.cat((input_tensor, attn_applied[0]), 1)
                decode_output = self.attn_combine(decode_output).unsqueeze(0)
                decode_output = F.relu(decode_output)
                decode_output,hn=self.lstm1(decode_output, (batch_hidden, batch_cell))
                decode_output,hn=self.lstm2(decode_output,hn)
                decode_output, (batch_hidden, batch_cell) = self.lstm3(decode_output, hn)
                decode_output = self.out(decode_output)
                decode_output = F.relu(decode_output)
                decode_outputs[i, j] = decode_output.view(-1)
                input_tensor=decode_output.view(1, -1)
        #decode_outputs=self.dropout(decode_outputs)
        return decode_outputs,attn

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            return result
        else:
            return result


class AttnDecoder3(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super(AttnDecoder3, self).__init__()
        self.hidden_size = hidden_size
        self.voca_size=vocab_size
        self.teacher_forcing_ratio=0.5
        self.stop=1

        self.conv2d=nn.Conv2d(in_channels=2*self.hidden_size,out_channels=2*self.hidden_size,kernel_size=1)
        self.hidden_linear = nn.Linear(self.hidden_size,2*self.hidden_size)
        self.input_linear = nn.Linear(self.hidden_size * 2, self.hidden_size//2)
        self.lstm1 = nn.LSTMCell(self.hidden_size//2, self.hidden_size)
        self.output_linear = nn.Linear(2*self.hidden_size,self.hidden_size)
        self.out = nn.Linear(self.hidden_size,self.voca_size)
        self.soft_max=nn.Softmax(dim=0)

    def forward(self, decode_inputs, decode_inputs_indexes, decode_hidden, encode_outputs):
        # decode_input=self.embed_dropout(decode_input)
        decode_batch_size, decode_length ,encode_length= decode_inputs.size()[0], decode_inputs.size()[1] ,encode_outputs.size()[1]
        #attn = encode_outputs.new_zeros(decode_batch_size,decode_length,encode_length)
        decode_outputs= decode_inputs_indexes.new_zeros(decode_inputs_indexes.size())
        loss_sum=0
        for i in range(decode_batch_size):
            batch_h0 = decode_hidden[0][i].unsqueeze(0)
            batch_c0 = decode_hidden[1][i].unsqueeze(0)
            batch_encode_output = encode_outputs[i]
            use_teacher_forcing = True
                # if random.random() < self.teacher_forcing_ratio and self.training == True else False
            batch_decode_input = decode_inputs[i]
            loss_batch = 0
            time_steps=0

            if use_teacher_forcing:
                for j in range(decode_length-2):
                    if j == 0:
                        # attention make
                        conv_encode_output = (
                            self.conv2d(batch_encode_output.permute(1, 0).unsqueeze(0).unsqueeze(3))).permute(0, 2, 3,
                                                                                                              1)
                        linear_state = self.hidden_linear(batch_h0)
                        # shape 1* s * h
                        tanh_state = torch.tanh(
                            conv_encode_output + linear_state.unsqueeze(0).unsqueeze(0).expand(1, encode_length, 1, -1))
                        # attn_ shape s
                        attn_weights = self.soft_max(
                            torch.sum(linear_state.expand(encode_length, -1) * tanh_state.squeeze(0).squeeze(1), 1))
                        # attn[i, j] = attn_weights
                        # value shape s*1*2h
                        attn_value = torch.sum(attn_weights.unsqueeze(1) * batch_encode_output, 0).unsqueeze(0)

                    input_tensor = batch_decode_input[j]
                    input_tensor = input_tensor.unsqueeze(0)+self.input_linear(attn_value)
                    # input_size is 1*0.5h,perhapes is to change the size of tensor ops

                    batch_h0, batch_c0=self.lstm1(input_tensor, (batch_h0, batch_c0))

                    # attention make
                    conv_encode_output = (
                        self.conv2d(batch_encode_output.permute(1, 0).unsqueeze(0).unsqueeze(3))).permute(0, 2, 3, 1)
                    linear_state = self.hidden_linear(batch_h0)
                    # shape 1* s * h
                    tanh_state = torch.tanh(
                        conv_encode_output + linear_state.unsqueeze(0).unsqueeze(0).expand(1, encode_length, 1, -1))
                    # attn_ shape s
                    attn_weights = self.soft_max(
                        torch.sum(linear_state.expand(encode_length, -1) * tanh_state.squeeze(0).squeeze(1), 1))
                    # attn[i, j] = attn_weights
                    # value shape s*1*2h
                    attn_value = torch.sum(attn_weights.unsqueeze(1) * batch_encode_output, 0).unsqueeze(0)

                    decode_output=batch_h0
                    decode_output=(self.output_linear(attn_value)+decode_output)

                    decode_output = self.out(decode_output)
                    decode_outputs[i,j+1]=decode_output.topk(1)[1].view(-1)
                    decode_output = F.log_softmax(decode_output,dim=1)

                    loss=F.nll_loss(decode_output,decode_inputs_indexes[i,j+1].view(-1))
                    loss_batch+=loss
                    time_steps=j+1
                    if decode_inputs_indexes[i,j+2] == self.stop:
                        break
            else:
                input_tensor = input_index[0].view(1, -1)
                for j in range(decode_length - 2):
                    attn_weights = F.softmax(self.attn(torch.cat((input_tensor, batch_hidden[0]), 1)), dim=1)
                    attn[i, j] = attn_weights[0]
                    attn_applied = torch.bmm(attn_weights.unsqueeze(0), batch_encode_output.unsqueeze(0))
                    decode_output = torch.cat((input_tensor, attn_applied[0]), 1)
                    decode_output = self.attn_combine(decode_output).unsqueeze(0)
                    decode_output = F.relu(decode_output)
                    decode_output, hn = self.lstm1(decode_output, (batch_hidden, batch_cell))
                    decode_output, hn = self.lstm2(decode_output, hn)
                    decode_output, (batch_hidden, batch_cell) = self.lstm3(decode_output, hn)
                    input_tensor=decode_output[0]
                    decode_output = self.out(decode_output).view(1,-1)
                    decode_output = F.log_softmax(decode_output,dim=0)
                    loss = F.nll_loss(decode_output, decode_input[i, j + 1].view(-1))
                    loss_batch += loss

                    if decode_inputs_indexes[i, j + 2] == self.stop:
                        break
            loss_batch /= time_steps
            loss_sum += loss_batch
        return loss_sum.view(1,-1),decode_outputs

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            return result
        else:
            return result


class TextData1(Dataset):
    def __init__(self, data, wv, train_len, label_len):
        self.data = data
        self.wv = wv
        self.train_len = train_len
        self.label_len = label_len
        self.dim = vec_dim

    def __len__(self):
        return len(self.data['titles'])

    def __getitem__(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization, the is used for classified model
        #        X = np.empty((self.batch_size, *self.train_dim, self.n_channels))
        #        y = np.empty((self.batch_size), dtype=int)
        # this is used for regression model
        encoder_input_data = torch.zeros(self.train_len, self.dim)
        decoder_input_data = torch.zeros(self.label_len, self.dim)
        decoder_target_data = torch.zeros(self.label_len, self.dim)

        contents, titles = self.data['contents'], self.data['titles']
        title = ['\t'] + titles[index]
        content = contents[index]
        length_temp=int(len(content)/3)
        #content=flatten(content)
        content_len = min(self.train_len, length_temp)
        title_len = len(title)

        content_vec = torch.from_numpy(self.wv[content[0:content_len]])
        title_vec = torch.from_numpy(self.wv[title])
        encoder_input_data[0:content_len] = content_vec
        decoder_input_data[0:title_len-1] = title_vec[0:title_len-1]
        decoder_target_data[0:title_len - 1] = title_vec[1:title_len]

        sample = {'encode_input': encoder_input_data, 'decode_input': decoder_input_data, 'target': decoder_target_data}
        return sample


class TextData2(Dataset):
    def __init__(self, data, dic, train_len, label_len):
        self.data = data
        self.dic = dic
        self.train_len = train_len
        self.label_len = label_len
        self.dim = vec_dim

    def __len__(self):
        return len(self.data['titles'])

    def __getitem__(self, index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization, the is used for classified model
        #        X = np.empty((self.batch_size, *self.train_dim, self.n_channels))
        #        y = np.empty((self.batch_size), dtype=int)
        # this is used for regression model
        encoder_input_data = torch.ones(self.train_len,dtype=torch.int64)
        decoder_input_data = torch.ones(self.label_len,dtype=torch.int64)

        contents, titles = self.data['contents'], self.data['titles']
        title = ['\t'] + titles[index]+['\n']
        content = ['\t']+contents[index]+['\n']
        #content=flatten(content)
        content_len = len(content)
        title_len = len(title)
        content.reverse()
        content_index=[self.dic.token2id[word] for word in content]
        title_index=[self.dic.token2id[word] for word in title]
        content_vec = torch.LongTensor(content_index)
        title_vec = torch.LongTensor(title_index)
        encoder_input_data[0:content_len] = content_vec
        decoder_input_data[0:title_len] = title_vec[0:title_len]

        sample = {'encode_input': encoder_input_data, 'decode_input': decoder_input_data}
        return sample


class RandomDataset1(Dataset):

    def __init__(self,max_encoder_seq_length):
        self.train_len = max_encoder_seq_length
        self.label_len=50
        self.dim=vec_dim

        self.encoder_input_data = torch.randn(1024,self.train_len, self.dim)
        self.decoder_input_data = torch.randn(1024,self.label_len, self.dim)
        self.decoder_target_data = torch.randn(1024,self.label_len, self.dim)

    def __getitem__(self, index):
        sample = {'encode_input': self.encoder_input_data[index], 'decode_input': self.decoder_input_data[index], 'target': self.decoder_target_data[index]}
        return sample

    def __len__(self):
        return self.encoder_input_data.size()[0]


class RandomDataset2(Dataset):

    def __init__(self,max_encoder_seq_length):
        self.train_len = max_encoder_seq_length
        self.label_len=50
        self.dim=vec_dim

        self.encoder_input_data = torch.ones((1024,self.train_len),dtype=torch.int64)
        self.decoder_input_data = torch.ones((1024,self.label_len),dtype=torch.int64)
        self.decoder_target_data = torch.ones((1024,self.label_len),dtype=torch.int64)

    def __getitem__(self, index):
        sample = {'encode_input': self.encoder_input_data[index], 'decode_input': self.decoder_input_data[index], 'target': self.decoder_target_data[index]}
        return sample

    def __len__(self):
        return self.encoder_input_data.size()[0]


class HistoryLoss():
    def __init__(self):
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def save(self, path):
        """losses,accuracy,val_loss,val_acc"""
        if os.path.exists(path):
            print('the file is exists!')
            os.remove(path)
        file=open(path,'w')
        json.dump(self.history,file)
        file.close()
        print('your history file has been saved!')

    def load(self,path):
        if os.path.exists(path):
            file = open(path, 'r')
            dict1=json.load(file)
            self.history=dict1
            print('your file has been read!')
        else:
            print('the file is not exists!')

    def loss_plot(self,loss_type='epoch'):
        iters = range(len(self.history['train_loss']))
        plt.figure()
        # acc
        plt.plot(iters, self.history['train_acc'], 'r', label='train acc')
        plt.plot(iters, self.history['val_acc'], 'b', label='val acc')
        # loss
        plt.plot(iters, self.history['train_loss'], 'g', label='train loss')
        plt.plot(iters, self.history['val_loss'], 'k', label='val loss')

        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(str(loss_type)+'.png')
#        plt.show()


def train_model(encoder, decoder, dataload, criterion,scheduler,history_loss,num_epochs=35,
                train_steps=512, val_steps=64, val_model='loss'):
    since = time.time()

    best_model_wts = 0
    best_acc = 10.0
    best_loss = 10.0

    if torch.cuda.device_count() > 1 and use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        encoder.cuda()
        decoder.cuda()
        print('model is on cuda by gpus!')
    elif torch.cuda.device_count()==1 and use_cuda:
        encoder.cuda()
        decoder.cuda()
        print('model is on cuda!')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0
        val_epoch_loss = 0.0
        val_epoch_acc = 0.0

        # 每一个迭代都有训练和验证阶段
        for phase in ['train', 'val']:
            steps = 0
            if phase == 'train':
                # scheduler.step()
                steps=train_steps
                scheduler.step()
                encoder.train(True)  # 设置 model 为训练 (training) 模式
                decoder.train(True)
            else:
                break
                steps=val_steps
                encoder.train(False)  # 设置 model 为评估 (evaluate) 模式
                decoder.train(False)

            epoch_loss = 0.0
            epoch_acc = 0.0

            # 遍历数据
            for i, data in enumerate(dataload[phase]):

                # 获取输入
                encode_inputs, decode_inputs = data['encode_input'], data['decode_input']

                # 用 Variable 包装输入数据
                if use_cuda:
                    if phase=='train':
                        inputs1 = Variable(encode_inputs.cuda())
                        inputs2 = Variable(decode_inputs.cuda())
                    elif phase=='val':
                        inputs1 = Variable(encode_inputs.cuda(),volatile=True)
                        inputs2 = Variable(decode_inputs.cuda(),volatile=True)
                        #labels = Variable(targets.cuda())
                else:
                    if phase == 'train':
                        inputs1 = Variable(encode_inputs)
                        inputs2 = Variable(decode_inputs)
                    elif phase == 'val':
                        inputs1 = Variable(encode_inputs, volatile=True)
                        inputs2 = Variable(decode_inputs, volatile=True)
                        # labels = Variable(targets.cuda())

                # 设置梯度参数为 0
                scheduler.optimizer.zero_grad()

                # 正向传递

                encode_outputs, encoder_hidden, decode_inputs = encoder(inputs1,inputs2)
                decode_hidden= encoder_hidden

                decode_outputs = decoder(decode_inputs, inputs2, decode_hidden, encode_outputs)
                loss=decode_outputs.sum()/batch_size
                #loss_tensor = criterion['loss'](decode_outputs, labels)
                #loss_ones=loss_tensor.new_ones(loss_tensor.size())
                #loss=((loss_ones-loss_tensor).sum())/batch_sizeaa
                #non_zeros=loss_tensor.nonzero().size()[0]
                #loss=(non_zeros-loss_tensor.sum())/non_zeros
                #acc = (criterion['acc'](decode_outputs, labels))
                acc=loss

                # 如果是训练阶段, 向后传递和优化
                if phase == 'train':
                    loss.backward()
                    scheduler.optimizer.step()
                    #encode_optimizer.step()
                    #decode_optimizer.step()

                # 统计
                epoch_loss += loss.data.item()
                epoch_acc += acc.data.item()
                #                running_corrects += torch.sum(preds == labels.data)

                if python_version == 2:
                    sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d} \r'.format(
                        phase, loss, acc,i+1,steps))
                    sys.stdout.flush()
                elif python_version == 3:
                    print('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d}'.format(
                        phase, loss, acc,i+1,steps),end='\r')

                if phase == 'train' and i == train_steps-1:
                    break
                elif phase == 'val' and i == val_steps-1:
                    break
            print('\n')

            if phase == 'train':
                train_epoch_loss = epoch_loss / train_steps
                train_epoch_acc = epoch_acc / train_steps
            #                epoch_acc=None

            elif phase == 'val':
                val_epoch_loss = epoch_loss / val_steps
                val_epoch_acc = epoch_acc / val_steps
            #               epoch_acc = None

            if train_epoch_loss < best_loss:
                best_loss = train_epoch_loss
                best_model_wts = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict()))
            else:
                print('model has not improving!')
            ''''# 深拷贝 model,两种模型正确率评价模式
            if val_model == 'acc' and phase == 'val':
                if val_epoch_acc < best_acc:
                    best_acc = val_epoch_acc
                    best_model_wts = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict()))
                else:
                    print('model has not improving!')
            elif val_model == 'loss' and phase == 'val':
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    best_model_wts = (copy.deepcopy(encoder.state_dict()), copy.deepcopy(decoder.state_dict()))
                else:
                    print('model has not improving!')'''

        # 保存最佳模型的权重
        torch.save(best_model_wts, path_base+'50k.1.best_model_wts')
        # 保存loss相关数据
        history_loss.history['train_loss'].append(train_epoch_loss)
        history_loss.history['train_acc'].append(train_epoch_acc)
        history_loss.history['val_loss'].append(val_epoch_loss)
        history_loss.history['val_acc'].append(val_epoch_acc)
        print('train: Loss: {:.4f} Acc: {:.4f} ;  val: Loss: {:.4f} Acc: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))
        print('\n')

    # 保存loss到文件
    history_loss.save(path_base+'50k.1.loss')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))


def evaluate(encoder, decoder, datasets):
    datasets=datasets['val']
    encoder.train(False)
    decoder.train(False)
    out=[]
    for i in range(100):
        sample=datasets[i]
        encode_inputs, decode_inputs = Variable(sample['encode_input'].unsqueeze(0).cuda()), Variable(sample['decode_input'].unsqueeze(0).cuda())
        encode_outputs, encoder_hidden, decode_inputs_vec = encoder(encode_inputs,decode_inputs)
        decode_hidden = encoder_hidden
        _, decode_outputs = decoder(decode_inputs_vec, decode_inputs, decode_hidden, encode_outputs)
        out.append(decode_outputs)
    return out


def test():
    start_dim = torch.randn(vec_dim)
    stop_dim = torch.randn(vec_dim)
    #if use_cuda:
       # stop_dim=stop_dim.cuda()
    encoder = Encoder3(voca_size=83888,embedd_size=128, hidden_size=256)
    decoder = AttnDecoder3(hidden_size=256, vocab_size=83888)
    #a=RandomDataset(max_encoder_seq_length)

    datasets = {'train': RandomDataset2(max_encoder_seq_length),
                'val': RandomDataset2(max_encoder_seq_length)}
    data_loads = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['train', 'val']}

    optimizer = optim.SGD([{'params': encoder.parameters(), 'lr': 0.01},
                           {'params': decoder.parameters(), 'lr': 0.01}], lr=0.01, momentum=0.9)

    lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.95 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2,lambda2])
    # scheduler1=optim.lr_scheduler.LambdaLR(optimizer1,lr_lambda=[lambda2])
    # scheduler2=optim.lr_scheduler.LambdaLR(optimizer2,lr_lambda=[lambda2])
    criterion = {'loss': nn.CosineSimilarity(dim=2), 'acc': nn.MSELoss()}
    loss_history=HistoryLoss()
    train_model(encoder, decoder, data_loads, criterion,scheduler,loss_history)


def main():
    from utils import FileIO
    from word_process import WordProcess
    # Path to the data txt file on disk.
    path_base = '../data/'
    path_file=path_base+'bytecup.corpus.train.0.50k.txt'
    fio = FileIO()
    word = WordProcess(path_base, is_model_load=False,is_dict_load=True)
    dic=word.dic
    contents,titles=fio.load_from_json(path_file)

    total_size = len(titles)
    num_samples = int(total_size * 0.8)
    num_test = total_size - num_samples
    print('num samples:',num_samples,'num tests:',num_test)

    max_encoder_seq_length = int(max([len(txt) for txt in contents]))+2
    max_decoder_seq_length = max([len(txt) for txt in titles]) + 2
    print('max_lengths:',max_encoder_seq_length,'  ',max_decoder_seq_length)

    train_data = {'contents': contents[0:num_samples], 'titles': titles[0:num_samples]}
    test_data = {'contents': contents[num_samples:total_size], 'titles': titles[num_samples:total_size]}
    datasets = {'train': TextData2(train_data, dic, train_len=max_encoder_seq_length, label_len=max_decoder_seq_length),
                'val': TextData2(test_data, dic, train_len=max_encoder_seq_length, label_len=max_decoder_seq_length)}
    data_loads = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=15) for x in ['train', 'val']}

    encoder = Encoder3(voca_size=84031,embedd_size=128, hidden_size=256)
    decoder = AttnDecoder3(hidden_size=256, vocab_size=84031)

    optimizer=optim.SGD([{'params':encoder.parameters(),'lr':0.01},
                         {'params':decoder.parameters(),'lr':0.01}],lr=0.01,momentum=0.9)

    lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.95 ** epoch
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda2,lambda2])
    criterion = {'loss': nn.CosineSimilarity(dim=2), 'acc': nn.MSELoss()}
    loss_history=HistoryLoss()
    train_model(encoder, decoder, data_loads, criterion, scheduler,loss_history)


def predict():
    from utils import FileIO,Utils
    from word_process import WordProcess

    # Path to the data txt file on disk.
    path_base = '../data/'
    path_file = path_base + 'bytecup.corpus.train.0.50k.txt'
    fio = FileIO()
    word = WordProcess(path_base, is_model_load=False,is_dict_load=True)

    contents, titles = fio.load_from_json(path_file)

    total_size = len(titles)
    num_samples = int(total_size * 0.8)
    num_test = total_size - num_samples
    print('num samples:', num_samples, 'num tests:', num_test)

    max_encoder_seq_length = int(max([len(txt) for txt in contents]))+2
    max_decoder_seq_length = max([len(txt) for txt in titles]) + 2
    print('max_lengths:',max_encoder_seq_length,'  ',max_decoder_seq_length)

    train_data = {'contents': contents[0:num_samples], 'titles': titles[0:num_samples]}
    test_data = {'contents': contents[num_samples:total_size], 'titles': titles[num_samples:total_size]}
    datasets = {'train': TextData2(train_data, word.dic, train_len=max_encoder_seq_length, label_len=max_decoder_seq_length),
                'val': TextData2(test_data, word.dic, train_len=max_encoder_seq_length, label_len=max_decoder_seq_length)}
    data_loads = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=15) for x in
                  ['train', 'val']}

    encoder = Encoder3(voca_size=84031, embedd_size=128, hidden_size=256)
    decoder = AttnDecoder3(hidden_size=256, vocab_size=84031)
    if use_cuda:
        encoder.cuda()
        decoder.cuda()
    best_model=torch.load(path_base+'./50k.1.best_model_wts')
    best_model=Utils().gpu_model_to_cpu_model(best_model)

    encoder.load_state_dict(best_model[0])
    decoder.load_state_dict(best_model[1])
    out=evaluate(encoder,decoder,datasets)

    file1=open(path_base+'50k.1.predict','a')
    for i,o in enumerate(out):
        file1.write(str([word.dic[int(i)] for i in o.data[0]]))
        file1.write(str(test_data['titles'][i])+'\n')
    file1.close()
    print('predict done!')


if __name__ == '__main__':
    predict()
