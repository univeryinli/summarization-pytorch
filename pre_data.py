import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords

from compiler.ast import flatten
import sys, os, json, pickle, multiprocessing
import numpy as np

from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from multiprocessing import Pool
import os, time, random
import Queue

def words2vectors(path_model, contents):
    model = Word2Vec.load(path_model)
    wv = model.wv
    del model
    print('model to vectors begin!')
    vectors = [wv[content] for content in contents]
    del wv
    print('words2vectors:' + path_model + ',done!')
    return vectors


def list_save(content, filename, mode='a'):
    # Try to save a list variable in txt file.
    print('listsave:' + filename + 'start!')
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()
    print('listsave:' + filename + 'done!')


def tex_save(content, filename):
    print('txtsave:' + filename + 'start!')
    file = open(filename)
    file.write(str(content), 'w')
    print('txtsave:' + filename + 'done!')
    file.close()


def list_read1(filename,path_model,flag=False):
    # Try to read a txt file and return a list.Return [] if there was a
    # mistake.
    try:
        file1 = open(filename, 'r')
    except IOError:
        error = []
        return error
    print('listread:' + filename + 'start!')
    model = Word2Vec.load(path_model)
    wv = model.wv
    del model
    print('model to vectors begin!')
    file2 = filename
    vectors=[]
    for line in file1:
        if flag:
            line = flatten(eval(line))
            vector = wv[line]
            vectors.append(vector)
        else:
            line = eval(line)
            vector = wv[line]
            vectors.append(vector)
    np.save(file2,vectors)
    file2.close()
    file1.close()
    print('listread:' + filename + 'done!')

model = Word2Vec.load(path_model)
wv = model.wv
del model

def list_read2(filename,flag=False):
    # Try to read a txt file and return a list.Return [] if there was a
    # mistake.
    try:
        file1 = open(filename, 'r')
    except IOError:
        error = []
        return error
    print('listread:' + filename + 'start!')
    file2 = filename
    vectors=[]
    for line in file1:
        if flag:
            line = flatten(eval(line))
            vector = wv[line]
            vectors.append(vector)
        else:
            line = eval(line)
            vector = wv[line]
            vectors.append(vector)
    np.save(file2,vectors)
    file2.close()
    file1.close()
    print('listread:' + filename + 'done!')

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


def wordtokenizer(sentence):
    #    toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    words = WordPunctTokenizer().tokenize(sentence)
    #    words=toker.tokenize(sentence)
    return words


def word2tokens(path, stoplist):
    file = open(path)
    contents = []
    titles = []
    title_contents = []
    for line in file.readlines():
        line = json.loads(line)
        id = line['id']
        print(id)
        content = line['content'].replace('\"', '"')
        content = content.replace(u'\u2014', '-').replace(u'\u201c', '"').replace(u'\u201d', '"').replace(u'\u2013',
                                                                                                          '-').replace(
            u'\u2018', "'").replace(u'\u2019', "'")
        content = content.lower()
        paras = content.split('\n')
        content = [
            [[word for word in wordtokenizer(sentence) if word not in stoplist] for sentence in splitSentence(para)] for
            para in paras]
        sentences = flatten(content)
        contents.append(content)
        title = line['title']
        title = title.lower()
        title = wordtokenizer(title)
        titles.append(title)
        title_content = sentences + title
        title_contents.append(title_content)
    file.close()
    return contents, titles, title_contents

def model_gen(path_model, title_content):
    model = Word2Vec(title_content, size=128, window=5, min_count=1, workers=100)
    model.save(path_model)
    print('model_gen:done')

path_base = '../data/'
path_source = path_base + 'corpus.txt'
path_train = path_base + 'train/'
path_label = path_base + 'label/'
path_title_con = path_base + 'title_content/'
path_model = path_base + 'title_content.models'
stoplist = ('\ , .  ( ) - ? / ! : ; !. [ ] + = _ " * ^ % #'.split())
stoplist = stopwords.words() + stoplist + ["'"]


# generate content and save file
# contents, titles, title_contents = word2tokens(path_source, stoplist)
# list_save(contents, path_train)
# list_save(titles, path_label)
# list_save(title_contents, path_title_con)
# fl = open(path_base + 'content.pickle', 'wb')
# pickle.dump(contents, fl,protocol=True)
# pickle.dump(titles, fl,protocol=True)
# pickle.dump(title_contents, fl,protocol=True)
# fl.close()

# generate and save model


def multu_process_file(path,path_model,IsFlatten=False):
    print('Parent process %s.', os.getpid())
    listdir = os.listdir(path)
    workers = len(listdir)-3
    prestr = listdir[0][:len(listdir[0]) - 2]
    print('workers:', workers)
    p = Pool(workers)
    #    res_list=[]
    for i in range(workers):
        pathread = path + prestr + str(i).zfill(2)
        print(pathread)
        p.apply_async(list_read, (pathread,path_model,IsFlatten))
        print('task:', i)
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')


#multu_process_file(path_train,path_model,IsFlatten=True)

# file1=open(path_base+'vectors_content.pickle','wb')
# pickle.dump(vectors_content,file1,protocol=True)
# file1.close()
# del vectors_content
print('vectors_content success!')

multu_process_file(path_label,path_model)
# file2=open(path_base+'vectors_title.pickle','wb')
# pickle.dump(vectors_title,file2,protocol=True)
# file1.close()
# del vectors_title
print('vectors_title success!')

# print(title_contents)
# model_gen(path_model, title_contents)
# model_gen(path_model2,title_content,contents)

path_vector_content = path_base + 'vectors_content.vec'
path_vector_title = path_base + 'vectors_title.vec'

print('done')
