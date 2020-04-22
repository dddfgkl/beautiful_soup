import os
import pandas as pd

"""
    DESC: 数据处理模块  
    Author: macong.ucaser@gmail.com  
"""

tsv_store_dir = "/home/machong/workspace/data/classification"

# 数据来源:https://github.com/xxxspy/Chinese_conversation_sentiment
# todo
def process_new_text_emotion():
    data_pos = "/home/machong/workspace/sentiment_analysis_textcnn/data/Chinese_conversation_sentiment"
    train_pos = os.path.join(data_pos, "sentiment_XS_30k.txt")
    valid_pos = os.path.join(data_pos, "sentiment_XS_test.txt")

    dataset = "Chinese_conversation"

    os.system(f"mkdir {tsv_store_dir}/{dataset}")

    train_data = pd.read_csv(train_pos, delimiter=",", encoding="utf-8")
    valid_data = pd.read_csv(valid_pos, delimiter=",", encoding="utf-8")

    # data = read_csv(chnsenticorp_pos, True)
    train_data = train_data.sample(frac=1)
    valid_data = valid_data.sample(frac=1)

    # dara = data.columns
    train_data.columns = ['labels', 'text']

    df1 = train_data
    df2 = valid_data

    print(df1['labels'][:10])

    df1['labels'] = df1['labels'].map(lambda x: 1 if str(x) == 'positive' else 0)

    print(len(df1))
    print(len(df2))
    df1.to_csv(f"{tsv_store_dir}/{dataset}/train.tsv", sep='\t', encoding='utf-8', index=False)
    df2.to_csv(f"{tsv_store_dir}/{dataset}/test.tsv", sep='\t', encoding='utf-8', index=False)

if __name__ == '__main__':
    process_new_text_emotion()