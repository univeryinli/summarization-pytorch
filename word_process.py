import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords
from iteration_utilities import flatten
import sys, os, json, pickle, multiprocessing,unidecode
import numpy as np

from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from multiprocessing import Pool
import os, time, random, gc,re
from utils import FileIO


class WordProcess:
    def __init__(self, path_base, is_model_load=False):
        self.path_base = path_base
        path_model = path_base + 'title_content.models'
        self.path_model = path_model
        path_train = path_base + 'train_1/'
        self.path_train = path_train
        path_label = path_base + 'label_1/'
        self.path_label = path_label
        self.re_sentence=re.compile(r'[\\\,\-\?\/\!\:\;\!\.\[\]\+\=\_\"\*\^\%\#\@\&\`\(\)\{\}\']+')
        self.re_num=re.compile(r'([0-9]+)')
        self.re_char=re.compile(r'[0-9]+')
        self.re_upper=re.compile(r'([A-Z][a-z]+)')
        self.content = []
        self.contents = []
        self.vector = []
        self.vectors = []

        if is_model_load:
            print('model load started!')
            model = Word2Vec.load(self.path_model)
            self.wv = model.wv
            del model
            print('model has been loaded!')

    def train_vec_manager(self, trainfile):
        """operate the files of the training data
            :param trainfile: the name of the trainfile like 'filename/'
            :return: no returns
        """
        path_train = self.path_base + trainfile
        listdir = os.listdir(path_train)
        print('train path manager begin!')
        for i in range(0, 5):
            for dir in listdir:
                if ('train' in dir) and (str(i).zfill(2) in dir):
                    filename = path_train + dir
                    self.content2vectors(filename)
        print('train path manager done')

    def label_vec_manager(self, labelfile):
        """operate the files of the training data
            :param labelfile: the name of the labelfile like 'filename/'
            :return: no returns
        """
        path_label = path_base + labelfile
        listdir = os.listdir(path_label)
        print('path manager begin!')
        for i in range(0, 5):
            for dir in listdir:
                if ('label' in dir) and (str(i).zfill(2) in dir):
                    filename = path_label + dir
                    self.title2vectors(filename)
        print('label path manager done')

    def text2tokens1(self, path, stop_list, is_return=False):
        stop_list_new = stopwords.words() + stop_list + ["'"]
        file = open(path)
        file_contents = open(path[0:-3] + 'contents.txt', 'a')
        file_titles = open(path[0:-3] + 'titles.txt', 'a')
        file_title_content = open(path[0:-3] + 'title_contents.txt', 'a')
        contents = []
        titles = []
        title_contents = []
        for line in file:
            line = json.loads(line)
            ids = line['id']
            print(ids)
            content = line['content'].replace('\"', '"')
            content = content.replace(u'\u2014', '-').replace(u'\u201c', '"').replace(u'\u201d', '"').replace(u'\u2013',
                                                                                                              '-').replace(
                u'\u2018', "'").replace(u'\u2019', "'")
            content = content.lower()
            paras = content.split('\n')
            content = [
                [[word for word in self.word_tokenizer(sentence) if word not in stop_list_new] for sentence in
                 self.split_sentence(para)]
                for
                para in paras]
            content = [[word for sentence in self.split_sentence(para) for word in self.word_tokenizer(sentence) if
                        word not in stop_list_new] for para in paras]
            sentences = flatten(content)
            title = line['title']
            title = title.lower()
            title = self.word_tokenizer(title)
            title_content = sentences + title
            if is_return:
                contents.append(content)
                titles.append(title)
                title_contents.append(title_content)
            else:
                file_contents.write(str(content))
                file_titles.write(title)
                file_title_content.write(title_content)
        file_title_content.close()
        file_titles.close()
        file_contents.close()
        file.close()
        if is_return:
            return contents, titles, title_contents

    def text2tokens2(self, path,is_return=False):
        file = open(path)
        file_new = open(path[0:-3] + 'new.txt', 'a')
        file_dic = open(path[0:-3] + 'dic.txt', 'a')
        file_title_contents = open(path[0:-3] + 'title_contents.txt', 'a')
        contents = []
        titles = []
        title_contents = []
        dic=['unt','\n','\t','SOS']
        for i,line in enumerate(file):
            content_new = {}
            text = json.loads(line)
            id = text['id']
            print(id)

            content = text['content']
            content = unidecode.unidecode(content)
            sentences = nltk.sent_tokenize(content)

            title = text['title']
            title = unidecode.unidecode(title)
            title = self.re_sentence.sub(' ', title).lower()
            title = nltk.word_tokenize(title)

            sentences_temp = []
            dic_temp=[]
            for j, sentence in enumerate(sentences):
                sentence = self.re_sentence.sub(' ', sentence)
                words = nltk.word_tokenize(sentence)
                words = [upper.lower() for word in words for char in self.re_num.split(word) if char is not '' for upper in
                         self.re_upper.split(char) if upper is not '']
                # re.match(r'[0-9]+[a-z]+', '2017at')
                sentences_temp.append(words)
                dic_temp += words
            title_content=dic_temp + title
            dic += list(set(title_content))

            # words = nltk.WordPunctTokenizer().tokenize('. How to Know If You ve Sent a Bad Twee.tA deep dive into    The Ratio ????')
            # print(words)

            if is_return:
                contents.append(sentences_temp)
                titles.append(title)
                title_contents.append(title_content)
            else:
                content_new['content']=sentences_temp
                content_new['title']=title
                content_new['id']=id
                content_new=json.dumps(content_new)
                file_new.write(content_new+'\n')
                file_title_contents.write(str(title_content)+'\n')
        file_title_contents.write(str(['unt','\n','\t','SOS']) + '\n')
        file_dic.write(str(list(set(dic)))+'\n')
        file_title_contents.close()
        file_new.close()
        file_dic.close()
        file.close()
        if is_return:
            return contents, titles, title_contents

    def content2vectors(self, path_train, is_return=False, is_saved=True):
        """can convert your lists of word like [['i'],['me']] to vectors
            :param path_train: lists of word like [['i'],['me']]
            :isReturn:
            :isSaved:
            :return: lists of vetors with the same shape of path_train
        """
        try:
            file = open(path_train, 'r')
        except IOError:
            error = []
            return error
        print('list read:' + path_train + 'start!')
        vectors = []
        for line in file:
            line = eval(line)
            line = flatten(line)
            con_size = len(line)
            #            print(con_size)
            pre_size = int(len(line) * 0.2)
            post_size = int(len(line) * 0.1)
            #    print(pre_size,post_size)
            content = []
            if con_size < 4:
                content = line
            elif 4 <= con_size < 10:
                content = line[0:2] + line[con_size - 1 - 1:con_size - 1]
            else:
                content = line[0:pre_size] + line[con_size - 1 - post_size:con_size - 1]
            #            print(con_size)
            vector = self.wv[content]
            vectors.append(vector)
        if is_return:
            return vectors
        elif is_saved:
            np.save(path_train, vectors)
        print('list read:' + path_train + 'done!')
        #    print(len(content),content)

    def title2vectors(self, path_label, is_return=False, is_saved=True):
        """can convert your lists of word like [['i'],['me']] to vectors
            :param path_label: lists of word like [['i'],['me']]
            :return: lists of vetors with the same shape of path_label
        """
        try:
            file = open(path_label, 'r')
        except IOError:
            error = []
            return error
        print('list read:' + path_label + 'start!')
        vectors = []
        for line in file:
            line = eval(line)
            vector = self.wv[line]
            vector = vector.flatten()
            vectors.append(vector)
        if is_return:
            return vectors
        elif is_saved:
            np.save(path_label, vectors)
        print('list read:' + path_label + 'done!')
        #    print(len(title),content)

    def model_gen(self, title_content):
        print('model_gen start!')
        model = Word2Vec(title_content, size=256, window=5, min_count=1, workers=15)
        model.save(self.path_model)
        print('model_gen:done')

    def cal_tfidf(self, title_contents, contents, titles):
        dictionary = corpora.Dictionary(title_contents)
        dictionary.save(path_base + 'bytecup.dict')  # store the dictionary, for future reference
        corpus_con = [dictionary.doc2bow(content) for content in contents]
        corpus_title = [dictionary.doc2bow(title) for title in titles]
        corpora.MmCorpus.serialize(path_base + 'bytecup_content.mm', corpus_con)  # store to disk, for later use
        corpora.MmCorpus.serialize(path_base + 'bytecup_title.mm', corpus_title)
        corpus_con = corpora.MmCorpus(path_base + 'bytecup_content.mm')
        corpus_title = corpora.MmCorpus(path_base + 'bytecup_title.mm')
        tfidf = models.TfidfModel(corpus_con)
        corpus_tfidf_con = tfidf[corpus_con]
        corpus_tfidf_title = tfidf[corpus_title]
        corpora.MmCorpus.serialize(path_base + 'bytecup_tfidf_con.mm', corpus_tfidf_con)
        corpora.MmCorpus.serialize(path_base + 'bytecup_tfidf_title.mm', corpus_tfidf_title)
        print('cal_tfidf:done!')

    def list_read(self,path,is_flatten=False,is_return=False):
        # Try to read a txt file and return a list.Return [] if there was a
        # mistake.
        try:
            file = open(path, 'r')
        except IOError:
            error = []
            return error
        print('list read:' + path + 'start!')
        file_lines=open(path+'dd','a')
        lines=[]
        for line in file:
            if is_flatten:
                line = flatten(eval(line))
            else:
                line = eval(line)
            if is_return:
                lines.append(line)
            else:
                file_lines.write(line)
        file_lines.close()
        file.close()
        print('list read:' + path + 'done!')
        if is_return:
            return lines

    def word_generator(self,path,is_flatten=False):
        file = open(path, 'r')
        print('generator read:' + path + 'start!')
        for line in file:
            if is_flatten:
                line = flatten(eval(line))
            else:
                line = eval(line)
            yield line
        file.close()
        print('generator:' + path + 'done!')


if __name__ == '__main__':
    path_base = '../data/'
    text_path=path_base + 'bytecup.corpus.train.0.txt'
    word_pro = WordProcess(path_base,is_model_load=True)
    #word_pro.text2tokens2(text_path)
    title_contents=word_pro.list_read(text_path[0:-3]+'title_contents.txt',is_return=True)
    word_pro.model_gen(title_contents)

#    word_pro.train_vec_manager('train_1/')
#    word_pro.label_vec_manager('label_1/')
