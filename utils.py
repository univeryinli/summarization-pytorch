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
            text=json.loads(line)
            content=text['content']
            content=list(flatten(content))
            content=content[0:int(len(content)*0.8)]
            contents.append(content)
            title=text['title']
            titles.append(title)
        file.close()
        return contents,titles


class Utils:
    def gpu_model_to_cpu_model(self, origin_state_dict):
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        def model_trans(state_dict):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v  # load params
            return new_state_dict

        if isinstance(origin_state_dict, tuple):
            return (model_trans(state) for state in origin_state_dict)
        elif isinstance(origin_state_dict, list):
            return [model_trans(state) for state in origin_state_dict]
        else:
            return model_trans(origin_state_dict)
