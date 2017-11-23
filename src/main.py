import json
import collections
from src.data import *
from src.config import *
import re
from matplotlib import pyplot as plt
import gensim

if __name__ == '__main__':
    f_in = open('../data/data.json')
    data = json.load(f_in)
    f_in.close()
    print(len(data))
    # print(data[0])
    data_set = {
        'train': [],
        'dev': [],
        'test': []
    }
    for i in data:
        data_set[i['set']].append(i)
    for k, v in data_set.items():
        print(k, len(v))

    word_count = 0
    for i in data:
        word_count += len([x for x in filter(lambda x: x[0] != '<', i['data'].split(' '))])
    print(word_count * 1.0 / len(data))

    # model = gensim.models.KeyedVectors.load_word2vec_format(Word2Vec, binary=True)
    #
    # print(model.word_vec('office'))
    # print(model.word_vec('test'))
    # print(model.word_vec('office'))

    for i in data_set['dev'][:1]:
        doc = i['data']
        print(doc)
        res = re.findall('<s>[^<]*</s>', doc)
        print(res)
        length = [len(_.split()) - 2 for _ in res]
        print(length)
        # plt.plot(length)
        # plt.show()


