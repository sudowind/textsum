import json

from keras import Input, Model
from keras.layers import MaxPool1D, Flatten, Dropout, Dense, GlobalMaxPooling1D, concatenate
from pythonrouge.pythonrouge import Pythonrouge
import re
import gensim
from src.config import *
import random
from keras.layers.convolutional import Conv1D, Conv2D
import numpy as np


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
        for i in data:
            self.data_set[i['set']].append(i)
        for k, v in self.data_set.items():
            print(k, len(v))

        f_in = open('../data/rouge_data.json')
        self.data_set = json.load(f_in)

        self.word2vec = json.load(open('word_vec.json'))

    @staticmethod
    def calc_item_rouge(item):
        """
        计算单个rouge
        :return:
        """
        para = item['data']
        # print(para)
        labels = item['label']
        pattern = '<s>([^<]*)</s>'
        sentences = re.findall(pattern, para)
        # print(len(sentences))
        reference = [re.findall(pattern, i)[0] for i in labels]
        # print(len(labels))
        all_sentences = sentences + reference
        ref = [[reference]]
        res = []
        for i in all_sentences:
            summary = [[i]]
            rouge = Pythonrouge(summary_file_exist=False,
                                summary=summary, reference=ref,
                                n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                                recall_only=True, stemming=True, stopwords=True,
                                word_level=True, length_limit=True, length=50,
                                use_cf=False, cf=95, scoring_formula='average',
                                resampling=True, samples=1000, favor=True, p=0.5)
            score = rouge.calc_score()
            print(i, score)
            res.append((i, score))
        return res

    def calc_rouge(self):
        """
        计算出所有的rouge值，并存储到文件
        :return:
        """
        count = 0
        for k in self.data_set.keys():
            for i in range(len(self.data_set[k])):
                res = self.calc_item_rouge(self.data_set[k][i])
                self.data_set[k][i]['rouge'] = res
                print(count)
                count += 1
        with open('../data/rouge_data.json', 'w+') as f_in:
            json.dump(self.data_set, f_in)

    def count_word(self, with_stop=False):
        """
        统计词的数量
        :return:
        """
        words = set()
        stop_words = set()
        if with_stop:
            with open('../stop-words.txt') as f_in:
                for i in f_in.readlines():
                    stop_words.add(i.strip())
        for data in self.data_set.values():
            for p in data:
                for w in re.split('[ -]', p['data']):
                    if with_stop:
                        if w not in stop_words:
                            words.add(w)
                    else:
                        words.add(w)
                for l in p['label']:
                    for w in re.split('[ -]', l):
                        if with_stop:
                            if w not in stop_words:
                                words.add(w)
                        else:
                            words.add(w)
        print(len(words))
        return list(words)
        # for i in enumerate(list(words)[:20]):
        #     print(i)

    def gen_word_vec(self, with_stop=False):
        """
        生成所有词的词向量
        :param with_stop:
        :return:
        """
        words = self.count_word(with_stop)
        out_name = 'word_vec_with_stop.json' if with_stop else 'word_vec.json'
        model = gensim.models.KeyedVectors.load_word2vec_format(Word2Vec, binary=True)
        vec = model.word_vec('afternoon')
        word_dict = dict()
        err_count = 0
        for i in range(len(words)):
            # print(words[i])
            # model.word_vec(words[i])
            try:
                word_dict[words[i]] = [_ for _ in model.word_vec(words[i])]
            except Exception as e:
                word_dict[words[i]] = [random.random() for i in range(300)]
                print(words[i])
                err_count += 1
            if i % 100 == 0:
                print(i)
        for i in word_dict.keys():
            total = sum(word_dict[i])
            word_dict[i] = [i / total for i in word_dict[i]]
        print('err_count: {}'.format(err_count))
        with open(out_name, 'w+') as f_out:
            json.dump(word_dict, f_out)
        # print(words[0], len(vec), model.word_vec(words[0]))

    def gen_sample(self):
        """
        生成训练样本
        :return:
        """
        para_len = 0
        max_para_len = 1000
        max_sen_len = 50
        para_len_list = []
        sen_len = 0
        sen_len_list = []
        sample = []
        for i in self.data_set['dev']:
            para = i['data']
            pattern = '<s>([^<]*)</s>'
            sentences = ' '.join(re.findall(pattern, para))
            # s['dev']
            # print(sentences)
            para_mat = []
            for w in re.split('[ -]', sentences):
                if w not in self.word2vec:
                    print(w)
                else:
                    para_mat.append(self.word2vec[w])
            while len(para_mat) < max_para_len:
                para_mat.append([0] * 300)
            para_mat = para_mat[:max_para_len]
            para_len = len(para_mat) if len(para_mat) > para_len else para_len
            para_len_list.append(len(para_mat))
            for s in i['rouge']:
                sen_mat = []
                rouge = s[1]['ROUGE-1']
                for w in re.split('[ -]', s[0]):
                    if w not in self.word2vec:
                        print(w)
                    else:
                        sen_mat.append(self.word2vec[w])
                while len(sen_mat) < max_sen_len:
                    sen_mat.append([0] * 300)
                sen_mat = sen_mat[:max_sen_len]
                sen_len = len(sen_mat) if len(sen_mat) > sen_len else sen_len
                sen_len_list.append(len(sen_mat))
                sample.append((np.array(para_mat), np.array(sen_mat), rouge))
        # print(para_len)
        # print(sen_len)
        # print(len(sample))
        # print(para_len_list)
        # print(sen_len_list)
        X_para = np.array([i[0] for i in sample])
        X_sen = np.array([i[1] for i in sample])
        Y = np.array([i[2] for i in sample])
        # print(X[0].shape)
        # print(X[1].shape)
        # print(X[2].shape)
        para_input = Input(shape=(1000, 300,))
        sen_input = Input(shape=(50, 300,))
        cnn1 = Conv1D(300, 3, padding='same', strides=1, activation='relu')(para_input)
        cnn1 = GlobalMaxPooling1D()(cnn1)

        cnn2 = Conv1D(300, 3, padding='same', strides=1, activation='relu')(sen_input)
        cnn2 = GlobalMaxPooling1D()(cnn2)

        all_input = concatenate([cnn1, cnn2])
        # flat = Flatten()(cnn1)
        # drop = Dropout(0.2)(flat)
        middle = Dense(64, activation='relu')(all_input)
        main_output = Dense(1, activation='relu')(middle)
        model = Model(inputs=[para_input, sen_input], outputs=main_output)
        model.compile(loss='mse', optimizer='sgd')
        model.fit([X_para, X_sen], Y,
                  batch_size=32,
                  epochs=15,
                  validation_data=([X_para, X_sen], Y))

