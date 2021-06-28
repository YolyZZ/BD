import random


def read_file(file_name):
    with open(file_name, encoding='U8') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


def read_corpus_file(file_name, label):
    '''
    :param file_name:
    :param label:
    :return: (text, label)
    '''
    with open(file_name, encoding='U8') as f:
        lines = f.readlines()
    lines = [(line.strip(), label) for line in lines]
    print('Label:{}\tLength:{}'.format(label, len(lines)))
    return lines


def read_items_file(file, sep='\t', item_index=-1):
    '''
    Load File And Split By sep(default '\t')
    :param file:
    :param sep:
    :param item_index:
    :return: (text, label, source)
    '''
    with open(file, encoding='U8') as f:
        lines = f.readlines()
    item_list = [line.strip().split(sep) for line in lines]
    if item_index != -1:
        item_list = [item[item_index] for item in item_list]
    return item_list


def save_file(data_list, filename):
    '''
    :param data_list: [text]
    :param filename:
    :return:
    '''
    data_list = ['{}\n'.format(data) for data in data_list]
    with open(filename, 'w', encoding='U8') as f:
        f.writelines(data_list)


# def save_label_source_file(data_list, filename):
#     '''
#     :param data_list: (text, label, source)
#     :param filename:
#     :return:
#     '''
#     data_list = ['{}\t{}\n'.format(data[0], data[1]) for data in data_list]
#     with open(filename, 'w', encoding='U8') as f:
#         f.writelines(data_list)


def save_label_source_file(data_list, filename):
    '''
    :param data_list: (text, label, source)
    :param filename:
    :return:
    '''
    data_list = ['{}\t{}\t{}\n'.format(data[0], data[1], data[2]) for data in data_list]
    with open(filename, 'w', encoding='U8') as f:
        f.writelines(data_list)


def save_items_file(data_list, filename):
    '''
    :param data_list: [items]
    :param filename:
    :return:
    '''
    data_list = ['{}\n'.format('\t'.join(data)) for data in data_list]
    with open(filename, 'w', encoding='U8') as f:
        f.writelines(data_list)


def filename_add_suffix(filename, suffix):
    items = filename.split('.')
    new_filename = '{}{}.{}'.format('.'.join(items[:-1]), suffix, items[-1])
    return new_filename


def build_train_dev_test_set(all_data, res_data_folder, need_shuffle=True):
    if need_shuffle:
        random.shuffle(all_data)

    print(len(all_data))
    train_size = int(len(all_data) * 0.8)
    eval_size = int(len(all_data) * 0.1)

    train_list = all_data[:train_size]
    dev_list = all_data[train_size:train_size + eval_size]
    test_list = all_data[train_size + eval_size:]

    save_label_source_file(train_list, res_data_folder + 'train.tsv')
    save_label_source_file(dev_list, res_data_folder + 'dev.tsv')
    save_label_source_file(test_list, res_data_folder + 'test.tsv')


def build_train_dev_set(all_data, res_data_folder, need_shuffle=True):
    if need_shuffle:
        random.shuffle(all_data)

    print(len(all_data))
    train_size = int(len(all_data) * 0.9)

    train_list = all_data[:train_size]
    dev_list = all_data[train_size:]

    save_label_source_file(train_list, res_data_folder + 'train.tsv')
    save_label_source_file(dev_list, res_data_folder + 'dev.tsv')
    save_label_source_file(all_data, res_data_folder + 'test.tsv')
