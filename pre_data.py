import json
import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer,WordPunctTokenizer
from compiler.ast import flatten
import sys,os
import numpy as np
from gensim import corpora, models, similarities
from gensim.models import Word2Vec

def list_save(content,filename,mode='a'):
    # Try to save a list variable in txt file.
    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()
    print('listsave:' + filename + ',done!')

def tex_save(content,filename):
    file = open(filename)
    file.write(str(content),'w')
    print('txtsave:'+filename+',done!')
    file.close()

def list_read(filename):
    # Try to read a txt file and return a list.Return [] if there was a
    # mistake.
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    lines = file.readlines()

    contents=[eval(line) for line in lines]
    file.close()
    print('listread:' + filename + ',done!')
    return contents

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences

def wordtokenizer(sentence):
#    toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    words = WordPunctTokenizer().tokenize(sentence)
#    words=toker.tokenize(sentence)
    return words

def word2tokens(path,stoplist):
    file=open(path)
    contents=[]
    titles=[]
    for line in file.readlines():
        line =json.loads(line)
        id=line['id']
        print(id)
        content=line['content'].replace('\"','"')
        content=content.replace(u'\u2014','-').replace(u'\u201c','"').replace(u'\u201d','"').replace(u'\u2013','-').replace(u'\u2018',"'").replace(u'\u2019',"'")
        content=content.lower()
        title=line['title']
        title=title.lower()
        title=wordtokenizer(title)
        titles.append(title)
        sentences=splitSentence(content)
        sentences_list=[]
        for sentence in sentences:
            words=wordtokenizer(sentence)
            words=[word.lower() for word in words if word not in stoplist]
            sentences_list=sentences_list+words
        contents.append(sentences_list)
        title_content=sentences_list+title
    file.close()
    return contents,titles,title_content

def model_gen(path_model,title_content):
    model = Word2Vec(title_content, size=128, window=5, min_count=1, workers=100)
    model.save(path_model)
    print('model_gen:done')

def words2vectors(path_model,contents):
    model=Word2Vec.load(path_model)
    vectors=model.wv
    print('model')
#    file=open(path_train)
#    print(type(file.readlines()[0]))
#    filesize=os.path.getsize(path_train)/3
#    filestr=''
#    for i in range(3):
#        filestr=file.read(filesize)
#    contents=eval(file.readlines()[0])
    vectors=[model.wv[content] for content in contents]
    print('words2vectors:'+path_model+',done!')
    return vectors

def cos_len(a,b):
    lena=np.sqrt(a.dot(a))
    lenb=np.sqrt(b.dot(b))
    coslen=a.dot(b)/(lena*lenb)
    angel=np.arccos(coslen)
    angel=angel*360/2/np.pi
    return angel

path_base='../data/'
path_source=path_base+'corpus.txt'
path_train=path_base+'train.data'
path_label=path_base+'label.data'
path_title_con=path_base+'title_content.data'
stoplist=set('\ , .  ( ) - ? / ! : ; !.'.split())
#generate content and save file
contents,titles,title_content=word2tokens(path_source,stoplist)
list_save(contents,path_train)
list_save(titles,path_label)
list_save(title_content,path_title_con)

path_model=path_base+'title_content.models'
model_gen(path_model,title_content)
#model_gen(path_model2,title_content,contents)

path_vector_content=path_base+'vectors_content.vec'
path_vector_title=path_base+'vectors_title.vec'

vectors_content= words2vectors(path_model,contents)
vectors_title= words2vectors(path_model,titles)

list_save(vectors_content,path_vector_content)
list_save(vectors_title,path_vector_title)

#print(model.similarity('love','loved'))
#for title in titles[0]:
#    print(model.wv[title.lower()])
#   print(model.similarity())

#dictionary = corpora.Dictionary(contents)
#print(dictionary.token2id)
#dictionary.save(path_base+'bytecup.dict')  # store the dictionary, for future reference
#corpus_con = [dictionary.doc2bow(content) for content in contents]
#corpus_title=[dictionary.doc2bow(title) for title in titles]
#corpora.MmCorpus.serialize(path_base+'bytecup_content.mm', corpus_con)  # store to disk, for later use
#corpora.MmCorpus.serialize(path_base+'bytecup_title.mm', corpus_title)
#corpus_con = corpora.MmCorpus(path_base+'bytecup_content.mm')
#corpus_title = corpora.MmCorpus(path_base+'bytecup_title.mm')

#tfidf = models.TfidfModel(corpus_con)
#corpus_tfidf_con = tfidf[corpus_con]
#corpus_tfidf_title = tfidf[corpus_title]
#corpora.MmCorpus.serialize(path_base+'bytecup_tfidf_con.mm', corpus_tfidf_con)
#corpora.MmCorpus.serialize(path_base+'bytecup_tfidf_title.mm', corpus_tfidf_title)

print('done')
