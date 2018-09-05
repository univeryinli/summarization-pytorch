import json
import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer
from compiler.ast import flatten
import sys,os
from gensim import corpora, models, similarities


def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences

from nltk.tokenize import WordPunctTokenizer
def wordtokenizer(sentence):
#    toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    words = WordPunctTokenizer().tokenize(sentence)
#    words=toker.tokenize(sentence)
    return words

def word2vector(path,stoplist):
    file=open(path)
    contents=[]
    titles=[]
    for line in file.readlines():
        line =json.loads(line)
        content=line['content'].replace('\n','.')
        title=line['title']
        title=wordtokenizer(title)
        titles.append(title)
        sentences=splitSentence(content)
        sentences_list=[]
        for sentence in sentences:
            words=wordtokenizer(sentence)
            words=[word.lower() for word in words if word not in stoplist]
            sentences_list=sentences_list+words
        contents.append(sentences_list)
    file.close()
    return contents,titles

path_base='../data/'
path=path_base+'bytecup.corpus.train.txt'
stoplist=set('\ , .  ( ) - ? / ! : ; !.'.split())
contents,titles=word2vector(path,stoplist)
dictionary = corpora.Dictionary(contents)
#print(dictionary.token2id)
dictionary.save(path_base+'bytecup.dict')  # store the dictionary, for future reference
corpus_con = [dictionary.doc2bow(content) for content in contents]
corpus_title=[dictionary.doc2bow(title) for title in titles]
corpora.MmCorpus.serialize(path_base+'bytecup_content.mm', corpus_con)  # store to disk, for later use
corpora.MmCorpus.serialize(path_base+'bytecup_title.mm', corpus_title)
corpus_con = corpora.MmCorpus(path_base+'bytecup_content.mm')
corpus_title = corpora.MmCorpus(path_base+'bytecup_title.mm')

tfidf = models.TfidfModel(corpus_con)
corpus_tfidf_con = tfidf[corpus_con]
corpus_tfidf_title = tfidf[corpus_title]
corpora.MmCorpus.serialize(path_base+'bytecup_tfidf_con.mm', corpus_tfidf_con)
corpora.MmCorpus.serialize(path_base+'bytecup_tfidf_title.mm', corpus_tfidf_title)

file=open(path_base+'train.data','w')
file.write(str(contents))
file.close()
file=open(path_base+'label.data','w')
file.write(str(titles))
file.close()
#file=open('./corpus','w')
#file.write(str(corpus))
#file.close()
