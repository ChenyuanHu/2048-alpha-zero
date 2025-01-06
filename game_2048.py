import numpy as np
import copy

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.is_player_turn = True  # True表示移动方向的玩家，False表示放置数字的玩家
        self.step = 0
        self.add_new_tile()
        self.add_new_tile()
        
    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.randint(len(empty_cells))]
            self.board[x, y] = 2 if np.random.random() < 0.9 else 4
            
    def get_state(self):
        return self.board.copy()

    def get_step(self):
        return self.step
    
    def get_score(self):
        return self.score
        
    def is_game_over(self):
        if np.any(self.board == 0):
            return False
        for i in range(4):
            for j in range(4):
                if i < 3 and self.board[i, j] == self.board[i + 1, j]:
                    return False
                if j < 3 and self.board[i, j] == self.board[i, j + 1]:
                    return False
        return True
    
    # 将(pos(x, y), value)转化为id线性动作空间
    def place_tile_id(self, vid):
        id = vid % 16
        value = 2 if id < 16 else 4
        """放置数字玩家的动作"""
        if self.board[id // 4, id % 4] == 0:
            self.board[id // 4, id % 4] = value
            self.is_player_turn = True
            return True
        return False
    
    def move(self, direction):
        """移动方向玩家的动作"""
        if not self.is_player_turn:
            return False
            
        # 0: up, 1: right, 2: down, 3: left
        original_board = self.board.copy()
        original_score = self.score
        
        # 统一转换为向左移动的情况
        if direction == 0:  # up
            self.board = np.rot90(self.board, k=1)
        elif direction == 1:  # right
            self.board = np.rot90(self.board, k=2)
        elif direction == 2:  # down
            self.board = np.rot90(self.board, k=3)
        
        # 对每一行进行处理
        for i in range(4):
            # 移除零并获取非零数字
            non_zero = self.board[i][self.board[i] != 0]
            
            # 合并相同的数字
            if len(non_zero) >= 2:
                result = []
                j = 0
                while j < len(non_zero):
                    if j + 1 < len(non_zero) and non_zero[j] == non_zero[j+1]:
                        merged = non_zero[j] * 2
                        self.score += merged
                        result.append(merged)
                        j += 2
                    else:
                        result.append(non_zero[j])
                        j += 1
                non_zero = np.array(result)
            
            # 用零填充到长度4
            new_line = np.zeros(4, dtype=int)
            if len(non_zero) > 0:
                new_line[:len(non_zero)] = non_zero
            self.board[i] = new_line
        
        # 旋转回原来的方向
        if direction == 0:  # up
            self.board = np.rot90(self.board, k=3)
        elif direction == 1:  # right
            self.board = np.rot90(self.board, k=2)
        elif direction == 2:  # down
            self.board = np.rot90(self.board, k=1)
            
        # 检查板子是否发生变化
        if not np.array_equal(original_board, self.board):
            self.is_player_turn = False
            self.step += 1
            return True
        else:
            self.score = original_score
            return False
            
    def get_max_tile(self):
        return np.max(self.board)
    
    def get_available_moves(self):
        """获取当前玩家的可用动作"""
        if self.is_player_turn:
            # 移动方向玩家的可用动作
            available_moves = []
            for direction in range(4):
                game_copy = copy.deepcopy(self)
                if game_copy.move(direction):
                    available_moves.append(direction)
            return available_moves
        else:
            # 放置数字玩家的可用动作
            empty_cells = self.get_empty_cells()
            available_moves = []
            for cell in empty_cells:
                id = cell[0] * 4 + cell[1]
                available_moves.append(id)
            for cell in empty_cells:
                id = cell[0] * 4 + cell[1] + 16
                available_moves.append(id)
            return available_moves
    
    def get_empty_cells(self):
        return list(zip(*np.where(self.board == 0)))
    
    def clone(self):
        return copy.deepcopy(self)
    
    def __str__(self):
        return str(self.board) 