import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords

from compiler.ast import flatten
import sys, os, json, pickle,multiprocessing
import numpy as np

from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from multiprocessing import Pool
import os, time, random

class MultiProcess():
    def __init__(self,workers):
        self.workers=workers

    def list_read(self):
        print('Parent process %s.', os.getpid())
        workers=self.workers
        print('workers:', workers)
        p = Pool(workers)
        for i in range(workers):
            pathread = path + prestr + str(i).zfill(2)
            print(pathread)
            p.apply_async(func,args)
            print('task:', i)
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')

    def list_read1(filename, path_model, flag=False):
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
        vectors = []
        for line in file1:
            if flag:
                line = flatten(eval(line))
                vector = wv[line]
                vectors.append(vector)
            else:
                line = eval(line)
                vector = wv[line]
                vectors.append(vector)
        np.save(file2, vectors)
        file2.close()
        file1.close()
        print('listread:' + filename + 'done!')