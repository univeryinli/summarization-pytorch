import os
from multiprocessing import Pool,cpu_count
from utils import utils


class multi_Process(utils):
    def __init__(self,workers=None):
        super.__init__()
        if workers is None:
            self.workers=cpu_count()
        else:
            self.workers=workers

    def long_time_task(self,name):
        print('Run task %s (%s)...', (name, os.getpid()))
        start = time.time()
        time.sleep(random.random() * 3)
        end = time.time()
        print('Task %s runs %0.2f seconds.', (name, (end - start)))
        return name

    def make_args_func(self,path,filesize=None):
        listdir = os.listdir(path)
        if filesize is None:
            filesize=len(listdir)
        for i in range(filesize):
            for dir in listdir:
                if ('npy' in dir) and (str(i).zfill(2) in dir):
                    pathread = path + dir
                    print(pathread)
                    yield (pathread,False,True)

    def multi_process(self,func_name,generator,isReturn=False):
        '''work with make make_args_func,func name
        you can use the utils`s list_read or list_write as your func_name
        :param func_name: list_read,list_write
        :param generator: make_args_func
        :param isReturn: return result or not return result
        :return: return the results of the processes
        '''

        print('Parent process %s.', os.getpid())
        print('workers:', self.workers)
        p = Pool(self.workers)
        res_list = []
        if isReturn:
            for i in range(self.workers):
                agrs=generator.next()
                res_list.append(p.apply_async(func_name,agrs))
                print('task:',i,'start!')
            print('Waiting for all subprocesses done...')
            p.close()
            p.join()
            print('merge processes results start!')
            results = []
            for i in range(len(res_list)):
                results = results + list(res_list[i].get())
                res_list[i] = 0
            del res_list
            gc.collect()
            print('All subprocesses done.')
            return results
        else:
            for i in range(self.workers):
                agrs=generator.next()
                p.apply_async(func_name,agrs)
                print('task:',i,'start!')
            print('Waiting for all subprocesses done...')
            p.close()
            p.join()
            print('All subprocesses done.')