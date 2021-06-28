# BD


1. requirement.txt：项目用到的packages及其版本参考


2. 程序入口：main.py
	内含三个API的入口，以及调用示例。请只修改main.py来适配输入格式。

1)senti_analysis_mass
	:param data_path: string, 民众舆论数据的绝对路径，数据结构见接口文档 输入表-民众
            event: string, 事件名称，twdx/byct
    :return senti_result: dataframe, 返回的结构化结果，对应接口文档输出表中的（1）-民众

2)senti_analysis_media
	:param data_path: string, 媒体舆论数据的绝对路径，数据结构见接口文档 输入表-媒体
            event: string, 事件名称，twdx/byct
    :return senti_result: dataframe, 返回的结构化结果，对应接口文档输出表中的（1）-媒体

3)statis_analysis
	:param df_mass: dataframe, senti_analysis_mass的输出，表示情感分析和关联分析之后的民众数据
            df_media: dataframe, senti_analysis_media的输出，表示情感分析和关联分析之后的媒体数据
            event: string, 事件名称，twdx/byct
    :return statis_res: dataframe, 返回的结构化结果，对应接口文档输出表中的（2）-态势预测
	

3. 调试步骤：

1)修改main.py对应API，修改读取数据的方式。
	目前是读取csv再转为dataframe类型，如果是dict格式，可以修改读入方式，根据数据文件内容改成读取dict并转为dataframe，保证转为了dataframe类型再传入底层逻辑就可以。

2)修改完毕后，运行main.py入口代码，可测试三个API的输出。

3)直接调用API，获得dataframe类型的输出，结构和内容详见接口文档。
