import numpy as np
import copy

class Game2048:
    def __init__(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.add_new_tile()
        self.add_new_tile()
        
    def add_new_tile(self):
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = empty_cells[np.random.randint(len(empty_cells))]
            self.board[x, y] = 2 if np.random.random() < 0.9 else 4
            
    def get_state(self):
        return self.board.copy()
    
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
    
    def move(self, direction):
        # 0: up, 1: right, 2: down, 3: left
        original_board = self.board.copy()
        original_score = self.score
        
        if direction in [0, 2]:  # up or down
            self.board = np.rot90(self.board)
        
        for i in range(4):
            line = self.board[i]
            if direction in [1, 2]:  # right or down
                line = line[::-1]
                
            # Remove zeros and merge identical numbers
            line = line[line != 0]
            for j in range(len(line) - 1, 0, -1):
                if line[j] == line[j-1]:
                    line[j] *= 2
                    self.score += line[j]
                    line[j-1] = 0
            line = line[line != 0]
            
            # Pad with zeros
            new_line = np.zeros(4, dtype=int)
            new_line[:len(line)] = line
            
            if direction in [1, 2]:  # right or down
                new_line = new_line[::-1]
            self.board[i] = new_line
            
        if direction in [0, 2]:  # up or down
            self.board = np.rot90(self.board, k=-1)
            
        # Check if the board changed
        if not np.array_equal(original_board, self.board):
            self.add_new_tile()
            return True
        else:
            self.score = original_score
            return False
            
    def get_max_tile(self):
        return np.max(self.board)
    
    def get_available_moves(self):
        available_moves = []
        for direction in range(4):
            game_copy = copy.deepcopy(self)
            if game_copy.move(direction):
                available_moves.append(direction)
        return available_moves
    
    def get_empty_cells(self):
        return list(zip(*np.where(self.board == 0)))
    
    def clone(self):
        return copy.deepcopy(self)
    
    def __str__(self):
        return str(self.board) 