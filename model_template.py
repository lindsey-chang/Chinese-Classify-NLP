import torch.nn as nn
import torch.nn.functional as F
import torch


class ClassifyModel(nn.Module):
    def __init__(self, vocab_size=2266, embed_dim=512, num_heads=8, num_classes=9, sequence_length=50):
        super(ClassifyModel, self).__init__()
        # 第一层是Embedding层
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 第二层是两个Transformer层
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

        # 第三层是两个CNN层
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5)

        # 第四层是两个LSTM层
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True)

        # 第五层是CLS层，这里假设是一个分类层
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # 通过Embedding层
        x = self.embedding(x)

        # 调整Tensor维度以匹配Transformer输入要求
        x = x.permute(1, 0, 2)  # 转换为 [seq_length, batch_size, embed_dim]
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # 转换回来

        # 通过两个CNN层
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, embed_dim, seq_length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 通过两个LSTM层
        x, (hn, cn) = self.lstm(x.permute(0, 2, 1))  # LSTM需要 [batch_size, seq_length, feature]

        # 选取LSTM最后一个时间步的输出进行分类
        x = self.fc(x[:, -1, :])

        return x


