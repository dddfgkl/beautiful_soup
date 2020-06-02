import os
import pandas as pd
import numpy as np
import pickle
import h5py

tsv_store_dir = "/home/machong/workspace/data/classification"
train_pos = os.path.join(tsv_store_dir, "Chinese_conversation", "train.tsv")
valid_pos = os.path.join(tsv_store_dir, "Chinese_conversation", "test.tsv")

h5_train_pos = os.path.join(tsv_store_dir, "Chinese_conversation", "h5train.pkl")
h5_valid_pos = os.path.join(tsv_store_dir, "Chinese_conversation", "h5valid.pkl")

# 建立词典
def build_vocab():
    train_data = pd.read_csv(train_pos, delimiter="\t", encoding="utf-8")
    valid_data = pd.read_csv(valid_pos, delimiter="\t", encoding="utf-8")
    train_store_file = open(h5_train_pos, 'wb+')
    valid_store_file = open(h5_valid_pos, 'wb+')

    vocab_count = {}
    max_length = 0

    # 统计词频
    for sen in train_data['text']:
        words = sen.strip().split()
        max_length = max(max_length, len(words))
        for word in words:
            if word in vocab_count.keys():
                vocab_count[word] += 1
            else:
                vocab_count[word] = 1

    for sen in valid_data['text']:
        words = sen.strip().split()
        max_length = max(max_length, len(words))
        for word in words:
            if word in vocab_count.keys():
                vocab_count[word] += 1
            else:
                vocab_count[word] = 1

    print("max length:", max_length)

    cnt = 2
    vocab_dict = {}
    vocab_dict['OOV'] = 0
    vocab_dict['PDD'] = 1
    for word in vocab_count.keys():
        # threshold set 2
        if vocab_count[word] >= 2:
            vocab_dict[word] = cnt
            cnt += 1

    # 处理train data，存储成h5形式的文件
    train_data_array = []
    train_label_array = []
    src_len = []
    for sen in train_data['text']:
        words = sen.strip().split()
        t_data = []
        src_len.append(len(words))
        for word in words:
            if word in vocab_dict:
                t_data.append(vocab_dict[word])
            else:
                t_data.append(vocab_dict['PDD'])
        if len(t_data) < max_length:
            t_data = np.pad(t_data, (1, max_length-len(t_data)), mode='constant')
        else:
            t_data = t_data[:max_length]
        train_data_array.append(t_data)

    train_label_array = np.array([x for x in train_data['labels']])
    train_data_array = np.array(train_data_array)
    src_len_array = np.array(src_len)

    train_file_pickle = {}
    train_file_pickle['data'] = train_data_array
    train_file_pickle['label'] = train_label_array
    train_file_pickle['sec_len'] = src_len_array
    train_file_pickle['word2cnt'] = vocab_dict
    train_file_pickle['cnt2word'] = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    pickle.dump(train_file_pickle, train_store_file)

    # 处理valid data，处理成h5文件的形式
    valid_data_array = []
    valid_label_array = []
    valid_src_len = []
    for sen in valid_data['text']:
        words = sen.strip().split()
        t_data = []
        valid_src_len.append(len(words))
        for word in words:
            if word in vocab_dict:
                t_data.append(vocab_dict[word])
            else:
                t_data.append(vocab_dict['OOV'])
        if len(t_data) < max_length:
            t_data = np.pad(t_data, (1, max_length-len(t_data)), mode="constant")
        else:
            t_data = t_data[:max_length]
        valid_data_array.append(t_data)

    valid_label_array = np.array([x for x in valid_data['labels']])
    valid_data_array = np.array(valid_data_array)
    valid_src_len_array = np.array(valid_src_len)

    valid_pickle = {}
    valid_pickle['data'] = valid_data_array
    valid_pickle['label'] = valid_label_array
    valid_pickle['src_len'] = valid_src_len_array
    valid_pickle['word2cnt'] = vocab_dict
    valid_pickle['cnt2word'] = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    pickle.dump(valid_pickle, valid_store_file)


def unit_test():
    build_vocab()

if __name__ == '__main__':
    unit_test()
