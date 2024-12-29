# 2048-AlphaZero

这是一个使用AlphaZero算法来训练2048游戏AI的项目。该项目使用深度强化学习和蒙特卡洛树搜索(MCTS)来学习2048游戏的最优策略。

## 项目结构

- `game_2048.py`: 2048游戏的核心实现
- `neural_network.py`: 神经网络模型定义
- `mcts.py`: 蒙特卡洛树搜索算法实现
- `train.py`: AI训练的主要逻辑

## 环境要求

- Python 3.11.9
- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- tqdm >= 4.65.0

## 安装说明

1. 克隆项目到本地：
```bash
git clone [your-repository-url]
cd 2048-alpha-zero
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 开始训练：
```bash
python train.py
```

训练过程会自动保存模型检查点和训练日志。你可以在 `training.log` 文件中查看训练进度。

## 特性

- 使用AlphaZero算法进行自我对弈训练
- 支持多进程并行自我对弈
- 实现了完整的2048游戏逻辑
- 使用PyTorch构建深度神经网络
- 支持训练过程的优雅终止
- 自动保存训练检查点

## 实现细节

- 使用深度卷积神经网络作为策略网络和价值网络
- 通过MCTS进行动作选择和策略改进
- 使用经验回放提高训练效率
- 支持模型的保存和加载 