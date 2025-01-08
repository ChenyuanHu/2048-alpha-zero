import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AlphaZeroNet(nn.Module):
    def __init__(self):
        super(AlphaZeroNet, self).__init__()
        # 输入层：4x4的游戏板，每个格子用16位one-hot编码表示(2^15=32768是2048游戏中最大可能的数)
        self.input_channels = 16
        
        # 共享特征提取层
        self.conv1 = nn.Conv2d(self.input_channels, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # 策略头（输出4个可能的移动方向的概率）
        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 4 * 4, 4)
        
        # 价值头（输出局面评估分数）
        self.value_conv = nn.Conv2d(128, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 4 * 4, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
    def _preprocess_input(self, state):
        # 将游戏板转换为one-hot编码
        # state shape: (batch_size, 4, 4)
        device = state.device
        batch_size = state.shape[0]
        x = torch.zeros((batch_size, self.input_channels, 4, 4), device=device)
        for i in range(self.input_channels):
            x[:, i] = (state == 2**(i+1)).float()
        return x
        
    def forward(self, state):
        # state shape: (batch_size, 4, 4)
        x = self._preprocess_input(state)
        
        # 共享特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

    def predict(self, state):
        """
        输入单个游戏状态，返回移动概率和价值评估
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            state_tensor = state_tensor.to(next(self.parameters()).device)
            policy, value = self(state_tensor)
            return policy.cpu().numpy()[0], value.cpu().numpy()[0][0] 