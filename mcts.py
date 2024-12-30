import numpy as np
import math
from game_2048 import Game2048
import torch
import os

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
            self.value = self.game_state.get_score() / 20000  # 归一化分数
        else:
            # 模型预测的策略和价值，策略是当前state下会走4个方向的概率，价值是当前局面的价值
            self.policy, self.value = self.model.predict(self.game_state.get_state())
            tmp = np.zeros(4)
            tmp[self.available_moves] = self.policy[self.available_moves]
            self.policy = tmp / tmp.sum()
        
    def select_child(self, c_puct=1.0):
        # AlphaZero的UCB变体，考虑先验概率
        s = sorted(self.children.items(),
                  key=lambda act_node: act_node[1].get_value() + 
                  c_puct * act_node[1].prior_probability * 
                  math.sqrt(self.number_of_visits) / (1 + act_node[1].number_of_visits))[-1]
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
        return self.value_sum / self.number_of_visits if self.number_of_visits > 0 else 0

class MCTS:
    def __init__(self, model, num_simulations=800, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def get_action_probs(self, game_state, temperature=1.0):
        root = MCTSNode(game_state=game_state.clone(), parent=None, parent_action=None, model=self.model)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection
            while node.untried_actions == [] and node.children != {}:
                action, node = node.select_child(self.c_puct)
            
            # Expansion
            # 如果还能够展开，如果无法展开，则node就为终局节点
            if node.untried_actions != []:
                # 根据策略网络的概率选择动作
                untried_probs = np.zeros(4)
                untried_probs[node.untried_actions] = node.policy[node.untried_actions]
                if untried_probs.sum() > 0:
                    untried_probs = untried_probs / untried_probs.sum()  # 重新归一化
                else:
                    # 如果所有概率都为0，使用均匀分布
                    untried_probs = np.ones(len(untried_probs)) / len(untried_probs)

                action = np.random.choice(np.arange(4), p=untried_probs)

                next_state = node.game_state.clone()
                next_state.move(action)
                node = node.expand(action, next_state, node.policy[action])
            
            value = node.value
            # Backup
            while node is not None:
                node.update(value)
                node = node.parent

        # 计算访问次数的概率分布
        visits = np.array([child.number_of_visits for child in root.children.values()])
        actions = np.array(list(root.children.keys()))
        
        if temperature == 0:
            # 选择访问次数最多的动作
            action = actions[np.argmax(visits)]
            probs = np.zeros(4)
            probs[action] = 1
            return action, probs
        else:
            # 根据temperature计算概率
            visits = visits ** (1/temperature)
            probs = np.zeros(4)
            probs[actions] = visits / visits.sum()
            action = np.random.choice(actions, p=visits/visits.sum())
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
    
    while not game.is_game_over():
        print(f"\nCurrent board:\n{game.board}")
        print(f"Current score: {game.get_score()}")
        
        # 使用MCTS选择最佳移动
        action, _ = mcts.get_action_probs(game, temperature=0.1)
        
        # 执行移动
        game.move(action)
        
    print("\nGame Over!")
    print(f"Final board:\n{game.board}")
    print(f"Final score: {game.get_score()}")
    print(f"Max tile: {game.get_max_tile()}")

if __name__ == "__main__":
    main() 