import torch
from sklearn.metrics import f1_score
from load_data import text_to_sequence,get_intent_index,get_char_index
import pandas as pd
from model_template import ClassifyModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def prepare_sequence(text_df, char_to_index, sequence_length=50):
    sequences = [text_to_sequence(text, char_to_index, sequence_length) for text in text_df['语料']]
    return torch.tensor(sequences, dtype=torch.long)


# 函数：测试模型并计算F1分数
def test_model_f1(model, test_loader, intent_to_index):
    model.eval()
    predictions = []
    true_intents = []

    with torch.no_grad():
        for sequences, intents in test_loader:
            sequences, intents = sequences.to(device), intents.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, dim=1)
            predictions.extend(predicted.cpu().numpy())
            true_intents.extend(intents.cpu().numpy())

    # 计算F1分数
    f1_macro = f1_score(true_intents, predictions, average='macro')
    f1_micro = f1_score(true_intents, predictions, average='micro')

    return f1_macro, f1_micro

train_df = pd.read_csv('./dataset/train.tsv', sep='\t')
test_df = pd.read_csv('./dataset/test.tsv', sep='\t')
val_df = pd.read_csv('./dataset/val.tsv', sep='\t')
combined_df=pd.concat([train_df,test_df,val_df],ignore_index=True)

char_to_index=get_char_index(combined_df)
intent_to_index=get_intent_index(train_df)
# 准备测试数据
test_sequences = prepare_sequence(test_df, char_to_index)
test_intents = torch.tensor([intent_to_index[intent] for intent in test_df['意图']], dtype=torch.long)

# 创建DataLoader
test_dataset = torch.utils.data.TensorDataset(test_sequences, test_intents)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 创建一个新的模型实例，这个实例必须和你保存的模型结构相同
model = ClassifyModel()
# 加载状态字典
model.load_state_dict(torch.load('./model/model_weights.pth', map_location=torch.device(device)))

# 测试模型并计算F1分数
f1_macro, f1_micro = test_model_f1(model, test_loader, intent_to_index)
print(f"F1 Macro: {f1_macro}, F1 Micro: {f1_micro}")
