import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer, WordPunctTokenizer
from nltk.corpus import stopwords
from iteration_utilities import flatten
import sys, os, json, pickle, multiprocessing,unidecode
from multiprocessing import Pool
import numpy as np

if os.name=='nt':
    import warnings
    warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim import corpora, models, similarities
from gensim.models import Word2Vec,TfidfModel
from multiprocessing import Pool
import os, time, random, gc,re
from utils import FileIO


class WordProcess:
    def __init__(self, path_base, is_model_load=False,is_dict_load=False,works=1):
        self.path_base = path_base
        self.path_model = path_base + 'title_content.models'
        self.path_dict =path_base+'50k.dict'
        self.path_text = path_base +'bytecup.corpus.train.0.50k.txt'
        self.works=works

        self.re_sentence=re.compile(r'[\\\,\-\?\/\!\:\;\!\.\[\]\+\=\_\"\*\^\%\#\@\&\`\(\)\{\}\']+')
        self.re_num=re.compile(r'([0-9]+)')
        self.re_char=re.compile(r'[0-9]+')
        self.re_upper=re.compile(r'([A-Z][a-z]+)')

        if is_model_load:
            print('model load started!')
            model = Word2Vec.load(self.path_model)
            self.wv = model.wv
            del model
            print('model has been loaded!')
        elif is_dict_load:
            print('dict load started!')
            dic=corpora.Dictionary().load(self.path_dict)
            self.dic=dic
            print('dict has been loaded started!')

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

    def text2tokens(self, path,is_return=False):
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

    def dic_gen(self):
        dictionary = Dictionary([['\t', '\n', 'SOS', 'STOP']])
        f = open(self., 'r')
        for line in f:
            text = json.loads(line)
            content = text['content']
            content = list(itertools.chain.from_iterable(content))
            title = text['title']
            title_content = content + title
            dictionary.add_documents([title_content])
        dictionary.save('50k.dict')

    def gen_new_dic(self,dict_path):
        dic = Dictionary().load(dict_path)
        dfs = dic.dfs
        dfs_new = sorted(dfs.items(), key=lambda dfs: dfs[1], reverse=True)
        dfs_new = dict(dfs_new)
        dfs_new_keys = list(dfs_new.keys())[0:70000]
        dic_vocab = [dic[i] for i in dfs_new_keys]
        return dic_vocab

    def cal_tfidf(self, contents,titles):
        print('cal tfidf:start!')
        corpus_con = [self.dic.doc2bow(content) for content in contents]
        tfidf = TfidfModel(corpus_con)
        contents_new=[]
        title_contents=[['\n','\t','SOS']]
        for content,title in zip(corpus_con,titles):
            dic=dict(tfidf[content])
            dic_new = dict(sorted(dic.items(), key=lambda dic: dic[1], reverse=True))
            dic_new_index=list(dic_new.keys())[0:min(200,len(dic_new))]
            content_new=[self.dic[index] for index in dic_new_index]
            contents_new.append(content_new)
            title_content=content_new+title
            title_contents.append(title_content)

        dictionary = corpora.Dictionary(title_contents)
        dictionary.save(path_base+'part_new.dict')  # store the dictionary, for future reference
        f=open(path_base+'content_new.txt','w')
        f.write(str(contents_new))
        f.close()
        print('cal tfidf:done!')

    def list_read(self,path,is_flatten=False,is_return=True):
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

    def vec2word(self,vec):
        vec=np.array(vec)
        word_t=[]
        for v in vec:
            word_t.append(self.wv.similar_by_vector(v)[0])
        return word_t

    def gen_new_text_by_new_vocab(self,vocab):
        f1 = open(self.path_base+'bytecup.corpus.train.0.full.txt', 'r')
        f2 = open(self.path_base+'bytecup.corpus.train.0.50k.txt', 'a')
        for i, line in enumerate(f1):
            print(i)
            text = json.loads(line)
            content = text['content']
            content_new = [[word if word in vocab else 'unk' for word in sen] for sen in content if 2 < len(sen) < 50]
            content_len = len(content_new)
            media_sen = [idx for i in range(0, content_len, max(1, int(content_len / 10))) for idx in
                         range(i, i + min(5, max(int(content_len / 10), 1))) if idx < content_len]
            sen_indexes = media_sen
            content_new = [content_new[i] for i in sen_indexes]
            text['content'] = content_new
            text = json.dumps(text)
            f2.write(text + '\n')
            i += 1
        f1.close()
        f2.close()

    def gen_new_text_by_tfidf(self):
        f=open(self.path_text,'r')
        tfidf=TfidfModel(dictionary=self.dic)
        f1=open(self.path_base+'bytecup.corpus.train.0.200tfidf.txt','a')
        for i,line in enumerate(f):
            print(i)
            text=json.loads(line)
            content=text['content']
            content=list(itertools.chain.from_iterable(content))
            content=[word for word in content if word not in stopword]
            content=self.dic.doc2bow(content)
            content=tfidf[content]
            content=dict(content)
            content_new = sorted(content.items(), key=lambda content: content[1], reverse=True)
            content_new_keys = list(dict(content_new).keys())[0:200]
            content_new=[self.dic[i] for i in content_new_keys]
            f1.write(str(content_new)+'\n')
        f1.close()
        f.close()

    def multi_list_read(self):
        print('Parent process %s.', os.getpid())
        workers = self.workers
        print('workers:', workers)
        p = Pool(workers)
        for i in range(workers):
            pathread = path + prestr + str(i).zfill(2)
            print(pathread)
            p.apply_async(func, args)
            print('task:', i)
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')


if __name__ == '__main__':
    path_base = '../data/'
    text_path=path_base + 'bytecup.corpus.train.0.txt'
    title_contents_path=path_base+'bytecup.corpus.train.0.title_contents.txt'
    word_pro = WordProcess(path_base,is_model_load=False,is_dict_load=True)
    #title_contents=word_pro.list_read(title_contents_path)
    #dic=word_pro.dic_gen(title_contents)
    #word_pro.text2tokens2(text_path)
    #title_contents=word_pro.list_read(text_path[0:-3]+'title_contents.txt',is_return=True)
    #word_pro.model_gen(title_contents)

#    word_pro.train_vec_manager('train_1/')
#    word_pro.label_vec_manager('label_1/')
