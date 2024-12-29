import numpy as np
import math
from game_2048 import Game2048
import torch

class MCTSNode:
    def __init__(self, game_state, parent=None, parent_action=None):
        self.game_state = game_state
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}
        self.number_of_visits = 0
        self.value_sum = 0
        self.prior_probability = 0
        self.untried_actions = game_state.get_available_moves()
        
    def select_child(self, c_puct=1.0):
        # AlphaZero的UCB变体，考虑先验概率
        s = sorted(self.children.items(),
                  key=lambda act_node: act_node[1].get_value() + 
                  c_puct * act_node[1].prior_probability * 
                  math.sqrt(self.number_of_visits) / (1 + act_node[1].number_of_visits))[-1]
        return s[0], s[1]
    
    def expand(self, action, game_state, prior_probability):
        node = MCTSNode(game_state=game_state, parent=self, parent_action=action)
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
        root = MCTSNode(game_state=game_state)
        
        for _ in range(self.num_simulations):
            node = root
            state = game_state.clone()
            
            # Selection
            while node.untried_actions == [] and node.children != {}:
                action, node = node.select_child(self.c_puct)
                state.move(action)
            
            # Expansion
            if node.untried_actions != []:
                # 获取神经网络的预测
                policy, value = self.model.predict(state.board)
                
                # 只考虑合法动作的概率
                legal_moves = state.get_available_moves()
                legal_probs = np.zeros(4)
                legal_probs[legal_moves] = policy[legal_moves]
                if legal_probs.sum() > 0:
                    legal_probs /= legal_probs.sum()
                
                action = np.random.choice(node.untried_actions)
                next_state = state.clone()
                next_state.move(action)
                node = node.expand(action, next_state, legal_probs[action])
            else:
                # 终局状态直接使用实际分数
                value = state.get_score() / 20000  # 归一化分数
            
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