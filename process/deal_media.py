import  pandas as pd
import numpy as np
from harvesttext import HarvestText
from harvesttext.resources import get_qh_sent_dict
from textrank4zh import TextRank4Keyword,TextRank4Sentence
from process import bert_classification_task
from process.utils import file_utils
import os

proj_path = os.path.abspath('.')

def get_event_keywords(file_path):
    keywords_data = []
    for x in open(file_path, 'r', encoding='utf-8'):
        x = x.strip('\n')
        keywords_data.append(x)
    return keywords_data

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]


def load_data(df, event):

    df_media = df
    # df_media.drop(columns=['location'], axis=1)
    # df_media = df_media.dropna()

    # deal timestamp data
    df_media['date'] = df_media['date'].apply(lambda x: x.split(' ')[0])
    df_media = df_media.sort_values(by='date')

    ht = HarvestText()
    # 清理文本内容为空的数据
    df_media = df_media.dropna(subset=['content'])
    # 文本内容归并、清洗、截短
    df_media['content'] = df_media['content'].replace(r'\n', ' ', regex=True)

    sents = df_media['content']

    # 重点人关联
    if event == "twdx":
        keywords_path = os.path.join(proj_path, 'data/keywords_tw.txt')
    elif event == "byct":
        keywords_path = os.path.join(proj_path, 'data/keywords_ct.txt')
    else:
        keywords_path = ''
    link_list = []
    for line in sents:
        entities_dict = ht.named_entity_recognition(line)
        name_list = get_keys(entities_dict, '人名')
        single_link = []
        if name_list != []:
            for name in get_event_keywords(keywords_path):
                y = name_list.count(name)
                if y != 0:
                    single_link.append(name + '-' + str(y))
        link_list.append(single_link)
    df_media["link_keywords"] = link_list
    # df_media["link_keywords"] = df_media["link_keywords"].map(lambda str: str[2:-2])

    # 文本截短
    df_media['content'] = df_media['content'].replace(r'\n', ' ', regex=True).apply(lambda x: x[0:200])

    # 输出bert预测数据集
    text_data_bert = pd.DataFrame({
        'x_test': df_media['content'].replace(r'\n', ' ', regex=True)
    })
    text_data_bert.to_csv(os.path.join(proj_path, "data_bert/mytest_media.tsv"), encoding='utf_8', index=False, sep='\t', header=False)

    # 初始化模型参数，输出预测结果
    res_file_path = os.path.join(proj_path, 'res_bert/sentiment_result_media.txt')  # 情感分析结果文件路径
    bert_classification_task.predict_contents_senti("mytest_media.tsv", res_file_path)

    # 载入bert结果路径
    class_res_file = file_utils.filename_add_suffix(res_file_path, '_class')
    df_media["sentiment"] = ''
    bert_senti_list = []
    with open(class_res_file, 'r', encoding='utf-8') as f:
        for line in f:
            bert_senti_list.append(int(line.strip('\n').split('\t')[1])-1)
    df_media["sentiment_bert"] = bert_senti_list

    # ht方法取值
    # sdict = get_qh_sent_dict()
    # sent_dict = ht.build_sent_dict(sents, min_times=1, pos_seeds=sdict["pos"], neg_seeds=sdict["neg"], scale="+-1")
    #
    # df_media["sentiment-1"] = ''
    # senti_list = []
    # df_media["sentiment_score"] = ''
    # score_list = []
    # for line in sents:
    #     res = '0'
    #     score = ht.analyse_sent(line)
    #     if score > 0.15:
    #         res = '1'
    #     elif score < -0.15:
    #         res = '-1'
    #     senti_list.append(res)
    #     score_list.append(score)
    # df_media["sentiment-1"] = senti_list
    # df_media["sentiment_score"] = score_list

    # 摘要summary：如有title则选取title，如没有title，则进行以下文本摘要
    # df_tw["summary"] = ''
    # summa_list = []
    # tr4s = TextRank4Sentence()
    # # tr4s.analyze(text=content, lower=True, source='all_filters')
    # for line in df_tw["content"]:
    #     # print("\nSentence:", line)
    #     tr4s.analyze(text=line, lower=True, source='all_filters')
    #     item = tr4s.get_key_sentences(num=1)[0]
    #     # print(item.index, item.weight, item.sentence)
    #     summa_list.append(item.sentence)
    # df_tw["summary"] = summa_list
    # tr4s = TextRank4Sentence()
    # df_tw["summary"] = df_tw["content"].apply(lambda x: tr4s.analyze(text=x, lower=True, source='all_filters').get_key_sentences(num=1)[0].sentence)
    # print(df_tw["summary"])

    new_data = pd.DataFrame(
        {'date': df_media['date'], 'source': df_media['source'],'url': df_media['url'], 'summary': df_media['title'],
         'content': df_media['content'],
         'forward': df_media['forward'],
         # 'sentiment_score': df_media["sentiment_score"],
         # 'sentiment-1': df_media['sentiment-1'],
         'sentiment': df_media["sentiment_bert"],
         'link_keywords': df_media["link_keywords"]
         })
    new_data.to_csv(os.path.join(proj_path, "res/media_tw_res.csv"), encoding='utf_8', index=False, sep=',')
    return new_data


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('max_colwidth', 200)
    # test
    df = pd.read_csv("../data/media_tw.csv", sep=',')
    senti_result = load_data(df, "twdx")
    print(senti_result)
