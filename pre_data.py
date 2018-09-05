import json
import nltk
import nltk.data
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from compiler.ast import flatten

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

contents=[]
titles=[]
with open('./train_sample.txt') as fl:
    for line in fl.readlines():
        line =json.loads(line)
        content=line['content'].replace('\n','.')
        title=line['title']
        title=wordtokenizer(title)
        titles.append(title)
        sentences=splitSentence(content)
        sentences_list=[]
        for sentence in sentences:
            words=wordtokenizer(sentence)
            sentences_list.append(words)
        contents.append(sentences_list)

stoplist=set('\ ,'.split())
#sentences=flatten(contents)
sentences=[]
for content in contents:
    sentences=sentences+content
dictionary = corpora.Dictionary(sentences)
print(dictionary.token2id)
dictionary.save('./bytecup.dict')  # store the dictionary, for future reference
corpus = [[dictionary.doc2bow(sentence) for sentence in content]for content in contents]
for corp in corpus:
    corpora.MmCorpus.serialize('./bytecup.mm', corp)  # store to disk, for later use
    corpus = corpora.MmCorpus('./bytecup.mm')
    print(corpus)
corpus = corpora.MmCorpus('./bytecup.mm')
#print(len(corpus[0]))
file=open('./train_sample','w')
file.write(str(contents))
file.close()
file=open('./lable_sample','w')
file.write(str(titles))
file.close()
file=open('./corpus','w')
file.write(str(corpus))
file.close()
