import json
from pythonrouge.pythonrouge import Pythonrouge


class DataGenerator(object):

    def __init__(self):
        f_in = open('../data/data.json')
        data = json.load(f_in)
        f_in.close()
        print(len(data))
        # print(data[0])
        self.data_set = {
            'train': [],
            'dev': [],
            'test': []
        }

    def generate_sample(self):
        """
        生成标记和训练样本
        :return:
        """
        pass

