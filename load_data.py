import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def text_to_sequence(text, char_to_index, sequence_length):
    # 转换文本为字符索引序列
    sequence = [char_to_index.get(char, char_to_index['<unk>']) for char in text]

    # 序列填充或截断
    if len(sequence) > sequence_length:
        sequence = sequence[:sequence_length]
    else:
        sequence += [char_to_index['<pad>']] * (sequence_length - len(sequence))

    return sequence


def get_intent_index(train_df):
    intent_to_index = {intent: index for index, intent in enumerate(train_df['意图'].unique())}
    return intent_to_index


def prepare_data_loaders(train_df, val_df, test_df, char_to_index, intent_to_index, sequence_length, batch_size=32):
    # 将文本转换为序列索引
    train_sequences = [text_to_sequence(row['语料'], char_to_index, sequence_length) for _, row in train_df.iterrows()]
    val_sequences = [text_to_sequence(row['语料'], char_to_index, sequence_length) for _, row in val_df.iterrows()]
    test_sequences = [text_to_sequence(row['语料'], char_to_index, sequence_length) for _, row in test_df.iterrows()]

    # 将意图转换为整数索引
    train_intents = [intent_to_index[row['意图']] for _, row in train_df.iterrows()]
    val_intents = [intent_to_index[row['意图']] for _, row in val_df.iterrows()]
    test_intents = [intent_to_index[row['意图']] for _, row in test_df.iterrows()]

    # 转换为适合模型输入的Tensor格式
    train_sequences_tensor = torch.tensor(train_sequences, dtype=torch.long)
    val_sequences_tensor = torch.tensor(val_sequences, dtype=torch.long)
    test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.long)

    train_intents_tensor = torch.tensor(train_intents, dtype=torch.long)
    val_intents_tensor = torch.tensor(val_intents, dtype=torch.long)
    test_intents_tensor = torch.tensor(test_intents, dtype=torch.long)

    # 创建TensorDataset和DataLoader
    train_dataset = TensorDataset(train_sequences_tensor, train_intents_tensor)
    val_dataset = TensorDataset(val_sequences_tensor, val_intents_tensor)
    test_dataset = TensorDataset(test_sequences_tensor, test_intents_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_char_index(combined_df):
    # 提取所有唯一字符
    unique_chars = set(''.join(combined_df['意图']) + ''.join(combined_df['语料']))

    # 创建字符到索引的映射
    char_to_index = {char: idx + 2 for idx, char in enumerate(unique_chars)}
    char_to_index['<pad>'] = 0  # 添加填充符
    char_to_index['<unk>'] = 1  # 添加未知字符

    return char_to_index
# # Assuming the CSV files are in the current working directory
# train_df = pd.read_csv('./dataset/train.tsv', sep='\t')
# test_df = pd.read_csv('./dataset/test.tsv', sep='\t')
# val_df = pd.read_csv('./dataset/val.tsv', sep='\t')
#
# print(f'Number of rows in train_df: {len(train_df)}')
# print(f'Number of rows in test_df: {len(test_df)}')
# print(f'Number of rows in val_df: {len(val_df)}')
#
# # 统计训练数据集中意图的种类数
# intent_types = train_df['意图'].nunique()
# # 统计train_df中“意图”每种种类的数量分布
# intent_distribution = train_df['意图'].value_counts()
# print(("train_df", intent_types, intent_distribution))
#
# # 统计意图的种类数
# intent_types_val = val_df['意图'].nunique()
# # 统计val_df中“意图”每种种类的数量分布
# intent_distribution_val = val_df['意图'].value_counts()
# print(("val_df", intent_types_val, intent_distribution_val))
#
# intent_types_test = test_df['意图'].nunique()
# intent_distribution_test = test_df['意图'].value_counts()
# print(("test_df", intent_types_test, intent_distribution_test))
#
# # 合并这三个数据集，为了创建词表
# combined_df=pd.concat([train_df,test_df,val_df],ignore_index=True)
#
# # 提取所有唯一字符
# unique_chars = set(''.join(combined_df['意图']) + ''.join(combined_df['语料']))
#
# # 创建字符到索引的映射
# char_to_index = {char: idx + 2 for idx, char in enumerate(unique_chars)}
# char_to_index['<pad>'] = 0  # 填充符
# char_to_index['<unk>'] = 1  # 未知字符
#
# # 查看部分映射
# print({k: char_to_index[k] for k in list(char_to_index)[:10]})
#
# # 根据字符到索引的映射确定词汇量大小
# vocab_size = len(char_to_index)  # 这里会包括所有唯一字符和特殊字符
# print(vocab_size)
