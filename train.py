import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import deque
import os
import shutil
from tqdm import tqdm
import logging
import multiprocessing as mp
import signal
import sys
from neural_network import AlphaZeroNet
from mcts import MCTS
from game_2048 import Game2048

# 全局变量用于控制训练终止
should_stop = False

def signal_handler(signum, frame):
    global should_stop
    print('\n正在优雅地停止训练...')
    should_stop = True

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

class GameMemory:
    def __init__(self, capacity=100000):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, policy, value):
        self.memory.append((state, policy, value))
        
    def extend(self, memories):
        self.memory.extend(memories)
        
    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        
        # 使用随机索引进行采样
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        samples = [self.memory[idx] for idx in indices]
        
        # 分离状态、策略和价值
        states = np.array([s[0] for s in samples])
        policies = np.array([s[1] for s in samples])
        values = np.array([s[2] for s in samples])
        
        return states, policies, values
    
    def __len__(self):
        return len(self.memory)

def play_game_worker(model_state_dict, device, num_simulations):
    try:
        # 创建模型副本
        model = AlphaZeroNet().to(device)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # 创建MCTS实例
        mcts = MCTS(model, num_simulations=num_simulations)
        
        # 玩一局游戏
        game = Game2048()
        states, policies, values = [], [], []
        
        while not game.is_game_over():
            state = game.board.copy()
            temperature = 1.0 if len(states) < 10 else 0.1
            action, policy = mcts.get_action_probs(game, temperature)
            
            states.append(state)
            policies.append(policy)
            game.move(action)
        
        final_score = game.get_score()
        normalized_score = final_score / 20000
        values = [normalized_score] * len(states)
        
        return states, policies, values, final_score
    except KeyboardInterrupt:
        return None

class ParallelSelfPlay:
    def __init__(self, model, num_workers, device, num_simulations):
        self.model = model
        self.num_workers = num_workers
        self.device = device
        self.num_simulations = num_simulations
        self.pool = None
        
    def init_pool(self):
        if self.pool is None:
            self.pool = mp.Pool(self.num_workers)
    
    def close_pool(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
        
    def play_games(self, num_games):
        global should_stop
        try:
            # 获取模型状态字典
            state_dict = self.model.state_dict()
            
            # 创建进程池
            self.init_pool()
            
            # 准备参数
            args = [(state_dict, self.device, self.num_simulations)] * num_games
            
            # 并行执行游戏
            results = []
            for result in tqdm(
                self.pool.starmap(play_game_worker, args),
                total=num_games,
                desc="Self-play games"
            ):
                if should_stop:
                    self.close_pool()
                    return None, None, None, None
                if result is not None:
                    results.append(result)
            
            # 整理结果
            all_states = []
            all_policies = []
            all_values = []
            all_scores = []
            
            for states, policies, values, score in results:
                all_states.extend(states)
                all_policies.extend(policies)
                all_values.extend(values)
                all_scores.append(score)
                
            return all_states, all_policies, all_values, all_scores
        except KeyboardInterrupt:
            self.close_pool()
            return None, None, None, None

def main():
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logging.info("Starting training")
    # 训练参数
    num_iterations = 1000    # 迭代次数, 包含Self-play的次数
    num_episodes = 20        # 每次迭代进行20次Self-play
    memory_capacity = 200000 # memory的容量
    num_batches = 100        # 每次迭代从memory中取数据进行训练的次数
    batch_size = 512         # 每次训练从memory中取batch size的数据进行训练
    num_simulations = 400    # MCTS的模拟次数
    num_workers = min(mp.cpu_count() - 2, 24)  # 留出2个核心给系统和训练进程
    
    # 设置设备和性能优化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # 启用自动混合精度训练
        scaler = torch.amp.GradScaler('cuda')
        # 启用cuDNN基准测试和TF32
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logging.info("Using CUDA device with mixed precision training")
    else:
        logging.info("Using CPU device")
    
    # 创建或加载模型
    model = AlphaZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    
    # 加载检查点（如果存在）
    if os.path.exists('checkpoints/checkpoint_latest.pt'):
        checkpoint = torch.load('checkpoints/checkpoint_latest.pt', weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iteration = checkpoint['iteration']
        if device.type == 'cuda' and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logging.info(f"Loaded checkpoint from iteration {start_iteration}")
    else:
        start_iteration = 0
    
    # 创建记忆库和并行自我对弈工作器
    memory = GameMemory(capacity=memory_capacity)
    parallel_self_play = ParallelSelfPlay(model, num_workers, device, num_simulations)
    
    try:
        # 训练循环
        for iteration in range(start_iteration, num_iterations):
            if should_stop:
                break
                
            model.eval()  # 自我对弈时使用评估模式
            # 并行自我对弈收集数据
            states, policies, values, scores = parallel_self_play.play_games(num_episodes)
            
            if should_stop or states is None:
                break
            
            # 更新记忆库
            for s, p, v in zip(states, policies, values):
                memory.push(s, p, v)
            
            # 记录分数
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            logging.info(f"Iteration {iteration}: Avg score = {avg_score:.0f}, Max score = {max_score:.0f}, Self-play actions = {len(states)}")
            
            # 训练网络
            model.train()  # 切换到训练模式
            policy_loss = 0
            value_loss = 0
            
            # 使用自动混合精度训练
            for _ in range(num_batches):
                states, policies, values = memory.sample(batch_size)
                states = torch.FloatTensor(states).to(device)
                policies = torch.FloatTensor(policies).to(device)
                values = torch.FloatTensor(values).to(device)
                
                if device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        # 前向传播
                        policy_pred, value_pred = model(states)
                        
                        # 计算损失
                        policy_loss_batch = -torch.sum(policies * torch.log(policy_pred + 1e-8)) / policies.size(0)
                        value_loss_batch = torch.mean((values - value_pred.squeeze()) ** 2)
                        total_loss = policy_loss_batch + value_loss_batch
                    
                    # 反向传播
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    policy_loss += policy_loss_batch.item()
                    value_loss += value_loss_batch.item()
                else:
                    # CPU训练
                    policy_pred, value_pred = model(states)
                    policy_loss_batch = -torch.sum(policies * torch.log(policy_pred + 1e-8)) / policies.size(0)
                    value_loss_batch = torch.mean((values - value_pred.squeeze()) ** 2)
                    total_loss = policy_loss_batch + value_loss_batch
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    
                    policy_loss += policy_loss_batch.item()
                    value_loss += value_loss_batch.item()
            
            policy_loss /= num_batches
            value_loss /= num_batches
            logging.info(f"Policy loss = {policy_loss:.4f}, Value loss = {value_loss:.4f}")
            
            # 保存检查点
            save_dict = {
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if device.type == 'cuda':
                save_dict['scaler_state_dict'] = scaler.state_dict()
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(save_dict, 'checkpoints/checkpoint_latest.pt')
            shutil.copy('checkpoints/checkpoint_latest.pt', f'checkpoints/checkpoint_iter{iteration + 1}.pt')
            logging.info(f"Saved checkpoint at iteration {iteration + 1}")
    
    except KeyboardInterrupt:
        logging.info("训练被用户中断")
    finally:
        # 确保资源被正确释放
        parallel_self_play.close_pool()
        logging.info("训练结束，资源已释放")

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn')
    main() 