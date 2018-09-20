import pickle, os,json
import numpy as np


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


class utils:
    def save(obj, path):
        if os.path.exists(path):
            os.remove(path)
            print('the file is exists!')
            file = open(path, 'wb')
            pickle.dump(obj, file, protocol=True)
            file.close()
            print('your file has been saved!')

    def load(path):
        if os.path.exists(path):
            file = open(path, 'rb')
            return pickle.load(file)
            print('your file has been readed!')
        else:
            print('the file is not exists!')
