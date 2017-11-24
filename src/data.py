import json
from pythonrouge.pythonrouge import Pythonrouge
import re


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

    @staticmethod
    def generate_sample(item):
        """
        生成标记和训练样本
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
            # print(score)
            res.append((i, score))
        return res

    def calc_rouge(self):
        """
        计算出所有的rouge值
        :return:
        """
        count = 0
        for k in self.data_set.keys():
            for i in range(len(self.data_set[k])):
                res = self.generate_sample(self.data_set[k][i])
                self.data_set[k][i]['rouge'] = res
                print(count)
                count += 1
        with open('../data/rouge_data.json', 'w+') as f_in:
            json.dump(self.data_set, f_in)

    def count_word(self):
        """
        统计词的数量
        :return:
        """
        words = set()
        for data in self.data_set.values():
            for p in data:
                for w in p['data'].split():
                    words.add(w)
                for l in p['label']:
                    for w in l.split():
                        words.add(w)
        print(len(words))
        for i in enumerate(list(words)[:20]):
            print(i)
