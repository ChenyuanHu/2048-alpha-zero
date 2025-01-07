import numpy as np
import math
from game_2048 import Game2048
import torch
import os
import graphviz
import logging

def normalize_score(score):
    return score / 2000000

class MCTSNode:
    def __init__(self, game_state, parent, parent_action, model, tile_action_size=2):
        self.model = model
        self.game_state = game_state
        self.available_moves = game_state.get_available_moves()
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.number_of_visits = 1
        self.value_sum = 0
        self.prior_probability = 0
        self.untried_actions = self.available_moves.copy()
        self.tile_action_size = tile_action_size   # 放置数字玩家的动作空间大小，减少计算量
        
        if self.available_moves == []:
            # 终局状态直接使用实际分数
            self.value = normalize_score(self.game_state.get_score())
            return

        # 模型预测的策略和价值
        if self.game_state.is_player_turn:
            self.policy, self.value = self.model.predict(self.game_state.get_state())
            # 移动方向玩家
            tmp = np.zeros(4)
            tmp[self.available_moves] = self.policy[self.available_moves]
            self.policy = tmp / (tmp.sum() + 1e-8)
            return

        # 放置数字玩家
        self.value = self.parent.value
        self.policy = np.array([9] * 16 + [1] * 16)
        if len(self.available_moves) > self.tile_action_size:
            self.available_moves = np.random.choice(self.available_moves, size=self.tile_action_size, replace=False).tolist()
            self.untried_actions = self.available_moves.copy()
        tmp = np.zeros(32)
        tmp[self.available_moves] = self.policy[self.available_moves]
        self.policy = tmp / (tmp.sum() + 1e-8)
        
    def select_child(self, c_puct=1.0, mcts=None):
        # AlphaZero的UCB变体，考虑先验概率
        def score(node):
            exploitation = node.get_value()
            n_parent = self.number_of_visits
            n_child = node.number_of_visits
            exploration = node.prior_probability * math.sqrt(n_parent) / (1 + n_child)
            exploration = c_puct * exploration
            
            # 记录统计信息
            if mcts is not None:
                mcts.stats['exploitation'].append(exploitation)
                mcts.stats['exploration'].append(exploration)
            
            return exploitation + exploration
        
        s = sorted(self.children.items(), key=lambda act_node: score(act_node[1]))[-1]
        return s[0], s[1]
    
    def expand(self, action, game_state, prior_probability):
        node = MCTSNode(game_state=game_state, parent=self, parent_action=action, model=self.model, tile_action_size=self.tile_action_size)
        node.prior_probability = prior_probability
        self.untried_actions.remove(action)
        self.children[action] = node
        return node
    
    def update(self, value):
        self.number_of_visits += 1
        self.value_sum += value
        
    def get_value(self):
        # 平均价值
        return self.value_sum / self.number_of_visits

    def to_dot(self, dot, node_id):
        # 创建当前节点的标签
        label = f"Action: {self.parent_action if self.parent_action is not None else 'Root'}\n"
        label += f"Visits: {self.number_of_visits}\n"
        label += f"Value: {self.value:.3f}\n"
        label += f"Score: {self.game_state.get_score()}\n"
        label += f"Board:\n{self.game_state.board}"
        if hasattr(self, 'policy'):
            label += f"\nPolicy: {np.array2string(self.policy, precision=2)}"
        
        # 添加节点
        dot.node(str(node_id), label)
        
        # 如果有父节点，添加边
        if self.parent is not None:
            parent_id = id(self.parent)
            dot.edge(str(parent_id), str(node_id))
        
        # 递归处理所有子节点
        for action, child in self.children.items():
            child.to_dot(dot, id(child))
        
        return dot

class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.0, visualization_dir=None, tile_action_size=2):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.visualization_dir = visualization_dir
        self.step_counter = 0
        self.root = None  # 保存搜索树的根节点
        self.node = None
        self.tile_action_size = tile_action_size
        # 添加统计对象
        self.stats = {
            'exploitation': [],
            'exploration': []
        }
        
    def reset_stats(self):
        self.stats = {
            'exploitation': [],
            'exploration': []
        }

    def print_stats(self):
        # 在返回之前输出统计信息
        if len(self.stats['exploitation']) > 0:
            for key in ['exploitation', 'exploration']:
                values = np.array(self.stats[key])
                logging.info(f"{key}: min={np.min(values):.4f}, max={np.max(values):.4f}, mean={np.mean(values):.4f}, std={np.std(values):.4f}, median={np.median(values):.4f}")
        
    def visualize_tree(self, root_node, step):
        """将MCTS树可视化为图像文件"""
        if self.visualization_dir is None:
            return
        
        dot = graphviz.Digraph(comment='MCTS Tree')
        dot.attr(rankdir='TB')
        
        # 从根节点开始构建图
        root_node.to_dot(dot, id(root_node))
        
        # 保存图像
        filename = os.path.join(self.visualization_dir, f'mcts_tree_step_{step}')
        dot.render(filename, view=False, format='svg')
        
    def get_action_probs(self, game_state, temperature=1.0):
        # 重置统计信息
        self.reset_stats()
        
        # 如果是新游戏或者根节点不存在，创建新的根节点
        if self.root is None:
            self.root = MCTSNode(game_state=game_state.clone(), parent=None, parent_action=None, model=self.model, tile_action_size=self.tile_action_size)
            self.node = self.root
        
        for _ in range(self.num_simulations):
            node = self.node
            
            # Selection
            while node.untried_actions == [] and node.children != {}:
                action, node = node.select_child(self.c_puct, None)
            
            # Expansion
            if node.untried_actions != []:
                # 根据策略网络的概率选择动作
                action_size = 4 if node.game_state.is_player_turn else 32
                
                # 计算未尝试动作的概率
                untried_probs = np.zeros(action_size)
                untried_probs[node.untried_actions] = node.policy[node.untried_actions]
                untried_probs = (untried_probs / untried_probs.sum() if untried_probs.sum() > 0 
                               else np.ones(action_size) / action_size)
                
                # 选择动作
                action = np.random.choice(np.arange(action_size), p=untried_probs)
                
                # 执行动作并扩展节点
                next_state = node.game_state.clone()
                if node.game_state.is_player_turn:
                    next_state.move(action)
                else:
                    next_state.place_tile_id(action)
                node = node.expand(action, next_state, node.policy[action])
            
            value = node.value
            # Backup
            while node != self.node:
                node.update(value)
                node = node.parent

        # 在选择动作之前生成可视化
        if self.visualization_dir:
            self.visualize_tree(self.root, self.step_counter)
            self.step_counter += 1

        # 计算访问次数的概率分布
        visits = np.array([child.number_of_visits for child in self.node.children.values()])
        actions = np.array(list(self.node.children.keys()))
        
        # 根据temperature计算概率
        visits = visits if temperature == 0 else visits ** (1/temperature)
        action_size = 4 if game_state.is_player_turn else 32
        
        # 选择动作
        if temperature == 0:
            action = actions[np.argmax(visits)]
        else:
            probs_visits = visits / visits.sum()
            action = np.random.choice(actions, p=probs_visits)
            
        # 计算概率分布
        probs = np.zeros(action_size)
        if temperature == 0:
            probs[action] = 1
        else:
            probs[actions] = visits / visits.sum()
            
        self.node = self.node.children[action]
        self.print_stats()
        return action, probs

def main():
    from neural_network import AlphaZeroNet
    
    model = AlphaZeroNet()
    if torch.cuda.is_available():
        model = model.cuda()

    # 加载检查点（如果存在）
    if os.path.exists('checkpoints/checkpoint_latest.pt'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load('checkpoints/checkpoint_latest.pt', weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded from checkpoint")
    
    game = Game2048()
    mcts = MCTS(model, num_simulations=4000, c_puct=1.0, tile_action_size=2)
    
    print(f"\nCurrent board:\n{game.board}")
    print(f"Current score: {game.get_score()}")
    while not game.is_game_over():
        action, _ = mcts.get_action_probs(game, temperature=0.1)
        if game.is_player_turn:
            # 移动方向玩家的回合
            game.move(action)
            last_action = action
        else:
            # 放置数字玩家的回合
            game.place_tile_id(action)
            # print(f"action: {action}")
            print(f"\nCurrent board:\n{game.board}")
            print(f"Current score: {game.get_score()}, action: {last_action}")
        
    print("\nGame Over!")
    print(f"Final board:\n{game.board}")
    print(f"Final score: {game.get_score()}")
    print(f"Max tile: {game.get_max_tile()}")

if __name__ == "__main__":
    main() 