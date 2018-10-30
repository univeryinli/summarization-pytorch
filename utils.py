import pickle, os,json
import numpy as np
from iteration_utilities import flatten


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class MyMath:
    def cos_len(self,a, b):
        lena = np.sqrt(a.dot(a))
        lenb = np.sqrt(b.dot(b))
        coslen = a.dot(b) / (lena * lenb)
        angel = np.arccos(coslen)
        angel = angel * 360 / 2 / np.pi
        return angel


class FileIO:
    def save2pickle(obj, path):
        if os.path.exists(path):
            os.remove(path)
            print('the file is exists!')
        file = open(path, 'wb')
        pickle.dump(obj, file, protocol=True)
        file.close()
        print('your file has been saved!')

    def load_from_pickle(self,path):
        if os.path.exists(path):
            file = open(path, 'rb')
            data=pickle.load(file)
            file.close()
            print('your file has been read!')
            return data
        else:
            print('the file is not exists!')

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

    def list_write(self,content, path):
        # Try to save a list variable in txt file.
        print('listsave:' + path + 'start!')
        file = open(path, 'a')
        for i in range(len(content)):
            file.write(str(content[i]) + '\n')
        file.close()
        print('list save:' + path + 'done!')

    def load_from_json(self,path):
        print('json file load start!')
        contents=[]
        titles=[]
        file = open(path, 'r')
        for line in file:
            line=json.loads(line)
            content=line['content']
            content=list(flatten(content))
            contents.append(content)
            title=line['title']
            titles.append(title)
        file.close()
        return contents,titles