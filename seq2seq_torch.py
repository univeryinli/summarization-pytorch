# -*- coding: utf-8 -*-
import torch, copy, time,sys
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import FileIO
from word_process import WordProcess
from iteration_utilities import flatten
import numpy as np
import torch.optim as optim


use_cuda = torch.cuda.is_available()
batch_size = 128
vec_dim=256

max_encoder_seq_length = 1000
max_decoder_seq_length=50


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
        encode_outputs=self.drop_out(encode_outputs)
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
    def __init__(self, input_size, hidden_size, batch_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=3, dropout=0.2)

    def forward(self, encode_inputs, encode_hidden):
        print('encode_input size:', encode_inputs.size())
        encode_outputs, encode_hidden = self.lstm(encode_inputs, encode_hidden)
        return encode_outputs, encode_hidden

    def init_hidden(self):
        result = Variable(torch.zeros(3, self.batch_size, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            results = (result, result)
            return results
        else:
            results = (result, result)
            return results


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
        decode_input=self.embed_dropout(decode_input)

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

                attn_weights = F.softmax(self.attn(torch.cat((input_tensor, batch_hidden[0]), 1)), dim=1)

                attn_applied = torch.bmm(attn_weights.unsqueeze(0), batch_encode_output.unsqueeze(0))
                decode_output = torch.cat((input_tensor, attn_applied[0]), 1)
                decode_output = self.attn_combine(decode_output).unsqueeze(0)

                decode_output = F.relu(decode_output)

                decode_output, batch_hidden = self.rnn(decode_output, batch_hidden)
                decode_output=self.out(decode_output)
                decode_outputs[i, j] = decode_output.view(-1)
        decode_outputs=self.dropout(decode_outputs)
        return decode_outputs

    def init_hidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            result = result.cuda()
            return result
        else:
            return result


class TextData(Dataset):
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


class RandomDataset(Dataset):

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


def train_model(encoder1, decoder1, dataload, criterion, encode_optimizer, decode_optimizer,scheduler1,scheduler2, num_epochs=25,
                train_steps=512, val_steps=64, val_model='loss'):
    since = time.time()

    best_model_wts = (encoder1.state_dict(), decoder1.state_dict())
    best_acc = 10.0
    best_loss = 10.0

    if torch.cuda.device_count() > 1 and use_cuda:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        encoder1 = nn.DataParallel(encoder1)
        decoder1 = nn.DataParallel(decoder1)
        encoder1.cuda()
        decoder1.cuda()
        print('model is on cuda by gpus!')
    elif torch.cuda.device_count()==1 and use_cuda:
        encoder1.cuda()
        decoder1.cuda()
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
                #                scheduler.step()
                steps=train_steps
                scheduler1.step()
                scheduler2.step()
                encoder1.train(True)  # 设置 model 为训练 (training) 模式
                decoder1.train(True)
            else:
                steps=val_steps
                encoder1.train(False)  # 设置 model 为评估 (evaluate) 模式
                decoder1.train(False)

            epoch_loss = 0.0
            epoch_acc = 0.0

            # 遍历数据
            for i, data in enumerate(dataload[phase]):

                # 获取输入
                encode_inputs, decode_inputs, targets = data['encode_input'], data['decode_input'], data['target']

                # 用 Variable 包装输入数据
                if use_cuda:
                    inputs1 = Variable(encode_inputs.cuda())
                    inputs2 = Variable(decode_inputs.cuda())
                    labels = Variable(targets.cuda())
                else:
                    inputs1, inputs2, labels = Variable(encode_inputs), Variable(decode_inputs), Variable(targets)

                # 设置梯度参数为 0
                encode_optimizer.zero_grad()
                decode_optimizer.zero_grad()

                # 正向传递

                encode_outputs, encoder_hidden = encoder1(inputs1)
                decode_hidden=encoder_hidden

                decode_outputs = decoder1(inputs2, decode_hidden, encode_outputs)

                loss_tensor = criterion['loss'](decode_outputs, labels)
                non_zeros=loss_tensor.nonzero().size()[0]
                loss=1-loss_tensor.sum()/non_zeros
                acc = (criterion['acc'](decode_outputs, labels))

                # 如果是训练阶段, 向后传递和优化
                if phase == 'train':
                    loss.backward()
                    encode_optimizer.step()
                    decode_optimizer.step()

                # 统计
                epoch_loss += loss.data.item()
                epoch_acc += acc.data.item()
                #                running_corrects += torch.sum(preds == labels.data)

                sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}  Step:{:3d}/{:3d} \r'.format(
                    phase, loss, acc,i+1,steps))
                sys.stdout.flush()

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

            # 深拷贝 model,两种模型正确率评价模式
            if val_model == 'acc' and phase == 'val':
                if val_epoch_acc < best_acc:
                    best_acc = val_epoch_acc
                    best_model_wts = (copy.deepcopy(encoder1.state_dict()), copy.deepcopy(decoder1.state_dict()))
                else:
                    print('model has not improving!')
            elif val_model == 'loss' and phase == 'val':
                if val_epoch_loss < best_loss:
                    best_loss = val_epoch_loss
                    best_model_wts = (copy.deepcopy(encoder1.state_dict()), copy.deepcopy(decoder1.state_dict()))
                else:
                    print('model has not improving!')

        print('train: Loss: {:.4f} Acc: {:.4f} ;  val: Loss: {:.4f} Acc: {:.4f}'.format(
            train_epoch_loss, train_epoch_acc, val_epoch_loss, val_epoch_acc))
        print('\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 保存最佳模型的权重
    torch.save(best_model_wts, './best_model_wts')


def test():
    start_dim = torch.randn(vec_dim)
    stop_dim = torch.randn(vec_dim)
    #if use_cuda:
       # stop_dim=stop_dim.cuda()
    encoder = Encoder1(input_size=256, hidden_size=256)
    decoder = AttnDecoder1(256, max_encoder_seq_length)
    a=RandomDataset(max_encoder_seq_length)


    datasets = {'train': RandomDataset(max_encoder_seq_length),
                'val': RandomDataset(max_encoder_seq_length)}
    data_loads = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=8) for x in ['train', 'val']}

    optimizer1 = optim.RMSprop(encoder.parameters())
    optimizer2 = optim.RMSprop(decoder.parameters())

    criterion = {'loss': nn.CosineSimilarity(dim=2), 'acc': nn.MSELoss()}
    train_model(encoder, decoder, data_loads, criterion, optimizer1, optimizer2)

'''
    #x1 = Variable(torch.randn(max_encoder_seq_length, batch_size, 256))
    #x2 = Variable(torch.randn(50, batch_size, 256))
    #label = Variable(torch.ones(50, batch_size, 256))

    hidden = encoder.init_hidden()
    encoder_out, h1 = encoder(x1, hidden)

    decode_out = decoder(x2, h1, encoder_out)

    optimizer1 = optim.SGD(encoder.parameters(), lr=0.03)
    optimizer2 = optim.SGD(decoder.parameters(), lr=0.01)
    criterion = {'loss': nn.CosineSimilarity(dim=2), 'acc': nn.MSELoss()}
    loss = torch.mean(criterion['loss'](decode_out, label))

    optimizer1.zero_grad()
    optimizer2.zero_grad()

    loss.backward()
    optimizer1.step()
    optimizer2.step()
'''


def main():
    # Path to the data txt file on disk.
    path_base = '../data/'
    path_file=path_base+'bytecup.corpus.train.0.new.txt'
    fio = FileIO()
    word = WordProcess(path_base, is_model_load=True)
    wv = word.wv
    start_dim = torch.from_numpy(wv['\t'])


    #contents = fio.list_read(path_base + 'bytecup.corpus.train.0.contents.txt', is_flatten=True, is_return=True)
    #titles = fio.list_read(path_base + 'bytecup.corpus.train.0.titles.txt', is_flatten=False, is_return=True)
    contents,titles=fio.load_from_json(path_file)

    total_size = len(titles)
    num_samples = int(total_size * 0.8)
    num_test = total_size - num_samples

    max_encoder_seq_length = int(max([len(txt) for txt in contents]) / 3)
    max_decoder_seq_length = max([len(txt) for txt in titles]) + 1

    train_data = {'contents': contents[0:num_samples], 'titles': titles[0:num_samples]}
    test_data = {'contents': contents[num_samples:total_size], 'titles': titles[num_samples:total_size]}
    datasets = {'train': TextData(train_data, wv, train_len=max_encoder_seq_length, label_len=max_decoder_seq_length),
                'val': TextData(test_data, wv, train_len=max_encoder_seq_length, label_len=max_decoder_seq_length)}
    data_loads = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=15) for x in ['train', 'val']}

    encoder = Encoder1(input_size=256, hidden_size=256)
    decoder = AttnDecoder1(256, max_encoder_seq_length)

    optimizer1 = optim.SGD(encoder.parameters(),lr=0.1,momentum=0.9)
    optimizer2 = optim.SGD(decoder.parameters(),lr=0.1,momentum=0.9)
    lambda1 = lambda epoch: epoch // 30
    lambda2 = lambda epoch: 0.95 ** epoch
    scheduler1=optim.lr_scheduler.LambdaLR(optimizer1,lr_lambda=[lambda2])
    scheduler2=optim.lr_scheduler.LambdaLR(optimizer2,lr_lambda=[lambda2])
    criterion = {'loss': nn.CosineSimilarity(dim=2), 'acc': nn.MSELoss()}
    train_model(encoder, decoder, data_loads, criterion, optimizer1, optimizer2,scheduler1,scheduler2)


if __name__ == '__main__':
    main()
