import pandas as pd
from process import deal_mass
from process import deal_media
from process import statistical_analysis



def senti_analysis_mass(data_path, event):
    '''
    :param data_path: string, 民众舆论数据的绝对路径，数据结构见接口文档输入表-民众
            event: string, 事件名称，twdx/byct
    :return senti_result: dataframe, 返回的结构化结果，对应接口文档输出表中的（1）-民众
    '''
    # 字典转为dataframe的一个示例：
    # dict_data = {}
    # with open('file_in.txt', 'r')as data:
    #     for line in data:
    #         for kv in [line.strip().split(':')]:
    #             dict_data.setdefault(kv[0], []).append(kv[1])
    #
    # columnsname = list(dict_data.keys())
    # df = pd.DataFrame(dict_data, columns=columnsname)

    # 转为dataframe再传入：
    df = pd.read_csv(data_path, sep=',')

    senti_result = deal_mass.load_data(df, event)
    return senti_result


def senti_analysis_media(data_path, event):
    '''
    :param data_path: string, 媒体舆论数据的绝对路径，数据结构见接口文档输入表-媒体
            event: string, 事件名称，twdx/byct
    :return senti_result: dataframe, 返回的结构化结果，对应接口文档输出表中的（1）-媒体
    '''
    # 字典转为dataframe的一个示例：
    # dict_data = {}
    # with open('file_in.txt', 'r')as data:
    #     for line in data:
    #         for kv in [line.strip().split(':')]:
    #             dict_data.setdefault(kv[0], []).append(kv[1])
    #
    # columnsname = list(dict_data.keys())
    # df = pd.DataFrame(dict_data, columns=columnsname)

    # 转为dataframe再传入：
    df = pd.read_csv(data_path, sep=',')

    senti_result = deal_media.load_data(df, event)
    return senti_result


def statis_analysis(df_mass, df_media, event):
    '''
    :param df_mass: dataframe, 调用senti_analysis_mass后的输出，表示情感分析和关联分析之后的民众数据
            df_media: dataframe, 调用senti_analysis_media后的输出，表示情感分析和关联分析之后的媒体数据
            event: string, 事件名称，twdx/byct
    :return statis_res: dataframe, 返回的结构化结果，对应接口文档输出表中的（2）-态势预测
    '''
    statis_res = statistical_analysis.predict_index(df_mass, df_media, event)
    return statis_res


if __name__ == '__main__':

    # 调用API示例：
    senti_mass_res = senti_analysis_mass("D:/yoly/BD/data/mass_tw.csv", "twdx")
    senti_media_res = senti_analysis_media("D:/yoly/BD/data/media_tw.csv", "twdx")
    statis_res = statis_analysis(senti_mass_res, senti_media_res, "twdx")

    print("-------------senti_mass_res-------------\n", senti_mass_res)
    print("-------------senti_media_res-------------\n", senti_media_res)
    print("-------------statis_res-------------\n", statis_res)




