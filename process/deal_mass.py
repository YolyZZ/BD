import pandas as pd
import numpy as np
from harvesttext import HarvestText
from harvesttext.resources import get_qh_sent_dict
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
    return [k for k, v in d.items() if v == value]


def load_data(df, event):
    # pd.set_option('display.max_columns', None)
    # pd.set_option('max_colwidth', 200)

    print("Project path: ", proj_path)
    df_mass = df

    # deal timestamp data
    df_mass['date'] = df_mass['date'].apply(lambda x: x.split(' ')[0]) # 时间只取date
    df_mass = df_mass.sort_values(by='date')

    # 用户标签的处理：
    # 影响力标签为粉丝多少，大于200为1，小于200为0（示例）（在确定用户标签前不做处理，直接传回具体粉丝数）
    df_mass['follow'] = df_mass['follow']
    # df_mass['follow'] = np.where(df_mass['follow'] > 200, 1, 0)

    ht = HarvestText()
    # 清理文本内容为空的数据
    df_mass = df_mass.dropna(subset=['content'])
    # 文本内容归并、清洗、截短
    df_mass['content'] = df_mass['content'].replace(r'\n', ' ', regex=True)

    sents = df_mass['content']

    # 重点人关联
    if event == "twdx":
        keywords_path = os.path.join(proj_path, 'data\keywords_tw.txt')
    elif event == "byct":
        keywords_path = os.path.join(proj_path, 'data\keywords_ct.txt')
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
    df_mass["link_keywords"] = link_list
    # df_mass["link_keywords"] = df_mass["link_keywords"].map(lambda str: str[2:-2])

    # 文本截短
    df_mass['content'] = df_mass['content'].replace(r'\n', ' ', regex=True).apply(lambda x: x[0:200])

    # 输出bert预测数据集
    text_data_bert = pd.DataFrame({
        'x_test': df_mass['content'].replace(r'\n', ' ', regex=True)
    })
    text_data_bert.to_csv(os.path.join(proj_path, 'data_bert/mytest_mass.tsv'), encoding='utf_8', index=False, sep='\t', header=False)

    # 初始化模型参数，输出预测结果
    res_file_path = os.path.join(proj_path, 'res_bert/sentiment_result_mass.txt') # 情感分析结果文件路径(概率)
    bert_classification_task.predict_contents_senti("mytest_mass.tsv", res_file_path)

    # 载入bert结果路径
    class_res_file = file_utils.filename_add_suffix(res_file_path, '_class') # 情感分析结果文件路径(分类)
    df_mass["sentiment"] = ''
    bert_senti_list = []
    with open(class_res_file, 'r', encoding='utf-8') as f:
        for line in f:
            bert_senti_list.append(int(line.strip('\n').split('\t')[1])-1)
    df_mass["sentiment_bert"] = bert_senti_list


    # ht方法取值
    # sdict = get_qh_sent_dict()
    # sent_dict = ht.build_sent_dict(sents, min_times=1, pos_seeds=sdict["pos"], neg_seeds=sdict["neg"], scale="+-1")
    #
    # df_mass["sentiment-1"] = ''
    # senti_list = []
    # df_mass["sentiment_score"] = ''
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
    # df_mass["sentiment-1"] = senti_list
    # df_mass["sentiment_score"] = score_list


    # 媒体关联
    if event == "byct":
        # 不做处理的小媒体list:
        # media_except = ['中华新闻', '中金在线', '中评网', '中青看点', '中商（zhongshang）民生头条网', '中外网', '中小企业河南网', '中亿财经网', '重庆时报社', '周口日报', '株洲日报', '紫牛新闻']
        df_mass['link_media'] = np.where(
            (df_mass['source'] == '微博') | (df_mass['source'] == '微信') | (df_mass['source'] == '懂车帝') | (
                        df_mass['source'] == '知乎'), 'media_big', 'media_small')
        df_mass.loc[
            (df_mass['source'] == '中华新闻') | (df_mass['source'] == '中金在线') | (df_mass['source'] == '中评网') | (df_mass['source'] == '中青看点')
            | (df_mass['source'] == '中商（zhongshang）民生头条网') | (df_mass['source'] == '中外网') | (df_mass['source'] == '中小企业河南网') | (
                    df_mass['source'] == '中亿财经网') | (df_mass['source'] == '重庆时报社') | (df_mass['source'] == '周口日报') | (
                    df_mass['source'] == '株洲日报') | (df_mass['source'] == '紫牛新闻'), 'link_media'] = ''
    elif event == "twdx":
        # 不做处理的小媒体list
        # media_except = ['星岛环球网', '新闻人网', '中华网', '一点号', '头条号', '乌有之乡网刊', '雪球', '网易']
        df_mass['link_media'] = np.where(
            df_mass['source'] == '微博', 'media_big', 'media_small')
        df_mass.loc[
            (df_mass['source'] == '星岛环球网') | (df_mass['source'] == '新闻人网') | (df_mass['source'] == '中华网') | (df_mass['source'] == '一点号')
            | (df_mass['source'] == '头条号') | (df_mass['source'] == '乌有之乡网刊') | (df_mass['source'] == '雪球') | (df_mass['source'] == '网易'), 'link_media'] = ''
    else:
        df_mass['link_media'] = np.where(
            df_mass['source'] == '微博', 'media_big', 'media_small')
        df_mass.loc[df_mass['source'] == '微信', 'link_media'] = ''


    new_data = pd.DataFrame({'date': df_mass['date'], 'url': df_mass['url'], 'source': df_mass['source'],'content': df_mass['content'],
                             'author': df_mass['author'], 'location': df_mass['location'], 'follow': df_mass['follow'],
                             # 'sentiment_score': df_mass["sentiment_score"],
                             # 'sentiment-1': df_mass['sentiment-1'],
                             'sentiment': df_mass["sentiment_bert"],
                             'link_keywords': df_mass["link_keywords"], 'link_media': df_mass["link_media"]
                             })
    new_data.to_csv(os.path.join(proj_path, 'res/mass_tw_res.csv'), encoding='utf_8', index=False, sep=',')
    return new_data



if __name__ == '__main__':
    # test
    df = pd.read_csv("../data/mass_tw.csv", sep=',')
    senti_result = load_data(df, "twdx")
    print(senti_result)