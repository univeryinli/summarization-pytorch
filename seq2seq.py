'''Sequence to sequence example in Keras (character-level).
This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.
# Summary of the algorithm
- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
# Data download
English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip
Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/
# References
- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense,advanced_activations
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
import numpy as np,random
from utils import FileIO
from word_process import WordProcess
from loss_history import LossHistory
from keras.layers.pooling import GlobalMaxPool1D
from data_generator import DataGenerator


def get_model(max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,latent_dim):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(max_encoder_seq_length, num_encoder_tokens))
    encoder_input1,state_h1,state_c1 = LSTM(latent_dim,return_sequences=True,return_state=True)(encoder_inputs)
    encoder_input2,state_h2,state_c2 = LSTM(latent_dim,return_sequences=True,return_state=True)(encoder_input1)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_input2)

    # We discard `encoder_outputs` and only keep the states.
    encoder_states1 = [state_h1, state_c1]
    encoder_states2 = [state_h2, state_c2]
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(max_decoder_seq_length,num_encoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.

    decoder_output1, _, _ = LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_inputs,initial_state=encoder_states)
    decoder_output2, _, _ = LSTM(latent_dim, return_sequences=True,return_state=True)(decoder_output1,initial_state=encoder_states2)
    decoder_output3, _, _ = LSTM(latent_dim,return_sequences=True,return_state=True)(decoder_output2,initial_state= encoder_states1)
    decoder_output3= Dense(num_encoder_tokens)(decoder_output3)(decoder_output3)
    decoder_outputs=advanced_activations.LeakyReLU(alpha=0.3)(decoder_output3)
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#    model = multi_gpu_model(model,gpus=2)
    # Run training
    model.compile(optimizer='rmsprop', loss='cosine',metrics=['mse'])
    return model


def batch_generator(data,batch_size,max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,wv,shuffle=True):
    count = 0
    if shuffle:
        print('shuffle start!')
        temp=[]
        for b_data in data:
            random.shuffle(b_data)
            temp.append(b_data)
        data=temp
        print('shuffle is done!')
    while True:
        start = count * batch_size
        end = (count + 1) * batch_size
        count += 1
        if (count + 1) * batch_size > len(data[0]):
            count = 0

        encoder_input_data = np.zeros(
            (batch_size, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = np.zeros(
            (batch_size, max_decoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_target_data = np.zeros(
            (batch_size, max_decoder_seq_length, num_encoder_tokens),
            dtype='float32')

        for i,(content,title) in enumerate(zip(data[0][start:end],data[1][start:end])):
            title=['\t']+title+['\n']
            content_vec = wv[content]
            title_vec = wv[title]
            encoder_input_data[i,0:len(content)]=content_vec[0:max_encoder_seq_length]
            decoder_input_data[i,0:len(title)]=title_vec
            decoder_target_data[i,0:len(title)-1]=title_vec[1:len(title)]

        yield ([encoder_input_data,decoder_input_data],decoder_target_data)

epochs = 50  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
GPUs=2
num_encoder_tokens=128
# Path to the data txt file on disk.
path_base = '../data1/'

# Vectorize the data.
fio=FileIO()
word=WordProcess(path_base,is_model_load=True)
wv=word.wv
contents=fio.list_read(path_base+'bytecup.corpus.train.0.contents.txt',is_flatten=True,is_return=True)
titles=fio.list_read(path_base+'bytecup.corpus.train.0.titles.txt',is_flatten=False,is_return=True)

total_size=len(titles)
num_samples=int(total_size*0.8)
num_test=total_size-num_samples
train_data=[contents[0:num_samples],titles[0:num_samples]]
test_data=[contents[num_samples:total_size],titles[num_samples:total_size]]

max_encoder_seq_length = int(max([len(txt) for txt in contents])/5)
max_decoder_seq_length = max([len(txt) for txt in titles])+2

params = { 'train_dim':max_encoder_seq_length,
           'label_dim':max_decoder_seq_length,
           'batch_size': 128,
           'n_classes': True,
           'n_channels': 128,
           'shuffle': True}

'''
input_texts = []
target_texts = []
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(contents))
print('Number of unique input tokens:', num_encoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
'''


print('Number of samples:', num_samples)
print('Number of samples:', num_test)
print('Number of unique input tokens:', num_encoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


model1=get_model(max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,latent_dim)
model1.summary()
train_generator=DataGenerator(train_data[0],train_data[1],wv, **params)
test_generator=DataGenerator(test_data[0],test_data[1],wv, **params)
#train_generator=batch_generator(train_data,batch_size1,max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,wv,shuffle=True)
#test_generator=batch_generator(test_data,batch_size2,max_encoder_seq_length,max_decoder_seq_length,num_encoder_tokens,wv,shuffle=True)
losshistory= LossHistory()
modelcheck=ModelCheckpoint(path_base+'bestmodel.m',monitor='loss',verbose=1,save_best_only=True)
model1.fit_generator(train_generator,steps_per_epoch=256,epochs=epochs,callbacks=[losshistory,modelcheck],validation_data=test_generator,validation_steps=10,use_multiprocessing=True,workers=8)
# Save model
losshistory.save('./history1.json')
losshistory.loss_plot('epoch')
losshistory.loss_plot('batch')
model1.save('s2s.h5')


'''
# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states


# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())



def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
'''
