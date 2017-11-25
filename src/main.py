import json
import collections
from src.data import *
from src.config import *
import re
from matplotlib import pyplot as plt
import gensim

if __name__ == '__main__':
    # f_in = open('../data/data.json')
    # data = json.load(f_in)
    # f_in.close()
    # print(len(data))
    # # print(data[0])
    # data_set = {
    #     'train': [],
    #     'dev': [],
    #     'test': []
    # }
    # for i in data:
    #     data_set[i['set']].append(i)
    # for k, v in data_set.items():
    #     print(k, len(v))
    #
    # word_count = 0
    # for i in data:
    #     word_count += len([x for x in filter(lambda x: x[0] != '<', i['data'].split(' '))])
    # print(word_count * 1.0 / len(data))

    # model = gensim.models.KeyedVectors.load_word2vec_format(Word2Vec, binary=True)
    #
    # print(model.word_vec('office'))
    # print(model.word_vec('test'))
    # print(model.word_vec('office'))

    generate = DataGenerator()
    # generate.calc_rouge()
    # generate.gen_word_vec()
    # generate.train_model()
    generate.test_model()

    # for i in data_set['dev'][:1]:
    #     doc = i['data']
    #     print(doc)
    #     res = re.findall('<s>[^<]*</s>', doc)
    #     print(res)
    #     length = [len(_.split()) - 2 for _ in res]
    #     print(length)
    #     generate.calc_item_rouge(i)
    #     # print(i['rouge'])
    #     # plt.plot(length)
    #     # plt.show()

    # generate.count_word()

    # summary = [[" Tokyo is the one of the biggest city in the world."]]
    # reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]
    #
    # # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
    # # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
    # # if recall_only=True, you can get recall scores of ROUGE
    # rouge = Pythonrouge(summary_file_exist=False,
    #                     summary=summary, reference=reference,
    #                     n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
    #                     recall_only=True, stemming=True, stopwords=True,
    #                     word_level=True, length_limit=True, length=50,
    #                     use_cf=False, cf=95, scoring_formula='average',
    #                     resampling=True, samples=1000, favor=True, p=0.5)
    # score = rouge.calc_score()
    # print(score)

