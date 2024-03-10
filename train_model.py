import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import copy
from load_data import prepare_data_loaders, get_intent_index
from model_template import ClassifyModel

train_df = pd.read_csv('./dataset/train.tsv', sep='\t')
test_df = pd.read_csv('./dataset/test.tsv', sep='\t')
val_df = pd.read_csv('./dataset/val.tsv', sep='\t')

# 合并这三个数据集，为了创建词表
combined_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

# 提取所有唯一字符
unique_chars = set(''.join(combined_df['意图']) + ''.join(combined_df['语料']))

# 创建字符到索引的映射
char_to_index = {char: idx + 2 for idx, char in enumerate(unique_chars)}
char_to_index['<pad>'] = 0  # 填充符
char_to_index['<unk>'] = 1  # 未知字符

# 查看部分映射
print({k: char_to_index[k] for k in list(char_to_index)[:10]})

# 根据字符到索引的映射确定词汇量大小
vocab_size = len(char_to_index)  # 这里会包括所有唯一字符和特殊字符

# 创建模型实例
# 参数包括词汇量大小、嵌入维度、头数、类别数和序列长度
vocab_size = 2266  # 假设的词汇量大小
embed_dim = 512  # 嵌入维度
num_heads = 8  # Transformer的头数
num_classes = 9  # 意图种类数
sequence_length = 50  # 序列长度，根据实际需要进行调整
# 假设 model 是已经定义的模型实例

model = ClassifyModel(vocab_size, embed_dim, num_heads, num_classes, sequence_length)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, epochs=1000,
                                    patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for sequences, intents in train_loader:
            sequences, intents = sequences.to(device), intents.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, intents)
            acc = accuracy(outputs, intents)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)

        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Check if the accuracy of the validation set has improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve == patience:
            print(f"No improvement in validation accuracy for {patience} consecutive epochs, stopping early...")
            break

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def validate_model(model, val_loader, criterion, device='cuda'):
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for sequences, intents in val_loader:
            sequences, intents = sequences.to(device), intents.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, intents)
            acc = accuracy(outputs, intents)

            val_loss += loss.item()
            val_acc += acc.item()

    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    return val_loss, val_acc


def save_model(path='.model/model_weights.pth'):
    torch.save(model.state_dict(), path)


intent_to_index = get_intent_index(train_df)
train_loader, val_loader, test_loader = prepare_data_loaders(train_df, val_df, test_df, char_to_index, intent_to_index,
                                                             sequence_length)
# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()

# 定义优化器为Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 执行训练和验证
best_model = train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, epochs=1000,
                                             patience=10)
