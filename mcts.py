import numpy as np
import math
from game_2048 import Game2048
import torch
import os
import graphviz

def normalize_score(score):
    return score / 2000000

class MCTSNode:
    def __init__(self, game_state, parent, parent_action, model):
        self.model = model
        self.game_state = game_state
        self.available_moves = game_state.get_available_moves()
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.number_of_visits = 0
        self.value_sum = 0
        self.prior_probability = 0
        self.untried_actions = self.available_moves.copy()
        
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
        if len(self.available_moves) > 2:
            self.available_moves = np.random.choice(self.available_moves, size=2, replace=False).tolist()
            self.untried_actions = self.available_moves.copy()
        tmp = np.zeros(32)
        tmp[self.available_moves] = self.policy[self.available_moves]
        self.policy = tmp / (tmp.sum() + 1e-8)
        
    def select_child(self, c_puct=1.0):
        # AlphaZero的UCB变体，考虑先验概率
        def score(node):
            exploitation = node.get_value()

            n_parent = self.number_of_visits
            n_child = node.number_of_visits
            exploration = node.prior_probability * math.sqrt(n_parent) / (1 + n_child)
            return exploitation + c_puct * exploration
        
        s = sorted(self.children.items(), key=lambda act_node: score(act_node[1]))[-1]
        return s[0], s[1]
    
    def expand(self, action, game_state, prior_probability):
        node = MCTSNode(game_state=game_state, parent=self, parent_action=action, model=self.model)
        node.prior_probability = prior_probability
        self.untried_actions.remove(action)
        self.children[action] = node
        return node
    
    def update(self, value):
        self.number_of_visits += 1
        self.value_sum += value
        
    def get_value(self):
        # 平均价值
        return self.value_sum / self.number_of_visits if self.number_of_visits > 0 else 0

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
    def __init__(self, model, num_simulations=800, c_puct=1.0, visualization_dir=None):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.visualization_dir = visualization_dir
        self.step_counter = 0
        self.root = None  # 保存搜索树的根节点
        self.node = None
        
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
        # 如果是新游戏或者根节点不存在，创建新的根节点
        if self.root is None:
            self.root = MCTSNode(game_state=game_state.clone(), parent=None, parent_action=None, model=self.model)
            self.node = self.root
        
        for _ in range(self.num_simulations):
            node = self.node
            
            # Selection
            while node.untried_actions == [] and node.children != {}:
                action, node = node.select_child(self.c_puct)
            
            # Expansion
            if node.untried_actions != []:
                # 根据策略网络的概率选择动作
                if node.game_state.is_player_turn:
                    # 移动方向玩家
                    untried_probs = np.zeros(4)
                    untried_probs[node.untried_actions] = node.policy[node.untried_actions]
                    if untried_probs.sum() > 0:
                        untried_probs = untried_probs / untried_probs.sum()
                    else:
                        untried_probs = np.ones(len(untried_probs)) / len(untried_probs)

                    action = np.random.choice(np.arange(4), p=untried_probs)         

                    next_state = node.game_state.clone()
                    next_state.move(action)
                    node = node.expand(action, next_state, node.policy[action])
                else:
                    # 放置数字玩家
                    untried_probs = np.zeros(32)
                    untried_probs[node.untried_actions] = node.policy[node.untried_actions]
                    if untried_probs.sum() > 0:
                        untried_probs = untried_probs / untried_probs.sum()
                    else:
                        untried_probs = np.ones(len(untried_probs)) / len(untried_probs)

                    action = np.random.choice(np.arange(32), p=untried_probs)         

                    next_state = node.game_state.clone()
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
        
        if temperature == 0:
            # 选择访问次数最多的动作
            action = actions[np.argmax(visits)]
            if game_state.is_player_turn:
                probs = np.zeros(4)
                probs[action] = 1
            else:
                probs = np.zeros(32)
                probs[action] = 1
            self.node = self.node.children[action]
            return action, probs
        else:
            # 根据temperature计算概率
            visits = visits ** (1/temperature)
            if game_state.is_player_turn:
                probs = np.zeros(4)
                probs[actions] = visits / visits.sum()
            else:
                probs = np.zeros(32)
                probs[actions] = visits / visits.sum()
            action = np.random.choice(actions, p=visits/visits.sum())
            self.node = self.node.children[action]
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
    mcts = MCTS(model)
    
    print(f"\nCurrent board:\n{game.board}")
    print(f"Current score: {game.get_score()}")
    while not game.is_game_over():
        action, _ = mcts.get_action_probs(game, temperature=0.1)
        if game.is_player_turn:
            # 移动方向玩家的回合
            game.move(action)
            print(f"\nCurrent board:\n{game.board}")
            print(f"Current score: {game.get_score()}, action: {action}")
        else:
            # 放置数字玩家的回合
            game.place_tile_id(action)
            print(f"action: {action}")
        
    print("\nGame Over!")
    print(f"Final board:\n{game.board}")
    print(f"Final score: {game.get_score()}")
    print(f"Max tile: {game.get_max_tile()}")

if __name__ == "__main__":
    main() 