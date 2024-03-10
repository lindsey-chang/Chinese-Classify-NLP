from model_template import ClassifyModel
import torch
from load_data import text_to_sequence,get_intent_index,get_char_index
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def nlp(text, model, char_to_index, intent_to_index, sequence_length=50):
    # 确保模型在评估模式
    model.eval()

    # 文本预处理并转换为序列索引
    sequence = text_to_sequence(text, char_to_index, sequence_length)
    sequence = torch.tensor([sequence], dtype=torch.long).to(device)

    # 进行预测
    with torch.no_grad():
        output = model(sequence)
        prediction = torch.argmax(output, dim=1)

    # 将预测的索引转换回意图标签
    index_to_intent = {v: k for k, v in intent_to_index.items()}
    predicted_intent = index_to_intent[prediction.item()]

    return predicted_intent

vocab_size = 2266  # 假设的词汇量大小
embed_dim = 512  # 嵌入维度
num_heads = 8  # Transformer的头数
num_classes = 9  # 意图种类数
sequence_length = 50  # 序列长度，根据实际需要进行调整

# 创建一个新的模型实例，这个实例必须和你保存的模型结构相同
model = ClassifyModel(vocab_size, embed_dim, num_heads, num_classes, sequence_length)

train_df = pd.read_csv('./dataset/train.tsv', sep='\t')
test_df = pd.read_csv('./dataset/test.tsv', sep='\t')
val_df = pd.read_csv('./dataset/val.tsv', sep='\t')
combined_df=pd.concat([train_df,test_df,val_df],ignore_index=True)

char_to_index=get_char_index(combined_df)
intent_to_index=get_intent_index(train_df)

# 加载状态字典
model.load_state_dict(torch.load('./model/model_weights.pth', map_location=torch.device(device)))

# 示例使用nlp函数进行预测
text = "你好，请推荐一个餐馆。"
predicted_intent = nlp(text, model, char_to_index, intent_to_index)
print(f"Predicted intent: {predicted_intent}")