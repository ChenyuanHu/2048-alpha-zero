import unittest
import numpy as np
from game_2048 import Game2048

class TestGame2048(unittest.TestCase):
    def setUp(self):
        self.game = Game2048()
    
    def test_player_turns(self):
        """测试玩家回合交替"""
        self.game.board = np.array([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.game.is_player_turn = True
        
        # 移动方向玩家移动后，应该切换到放置数字玩家
        self.game.move(2)  # 向下移动
        self.assertFalse(self.game.is_player_turn)
        
        # 放置数字玩家放置后，应该切换到移动方向玩家
        self.game.place_tile((0, 1), 2)
        self.assertTrue(self.game.is_player_turn)
        
        # 无效的移动不应该切换玩家
        self.game.board = np.array([
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.game.is_player_turn = True
        result = self.game.move(0)  # 向上移动，无效
        self.assertTrue(result == False)
        self.assertTrue(self.game.is_player_turn)
        
        # 无效的放置也不应该切换玩家
        self.game.is_player_turn = False
        result = self.game.place_tile((0, 0), 2)  # 已经有数字的位置
        self.assertTrue(result == False)
        self.assertFalse(self.game.is_player_turn)
    
    def test_available_moves(self):
        """测试获取可用动作"""
        # 测试移动方向玩家的可用动作
        self.game.board = np.array([
            [2, 0, 0, 0],
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.game.is_player_turn = True
        moves = self.game.get_available_moves()
        self.assertTrue(set(moves).issubset({0, 1, 2, 3}))
        self.assertIn(2, moves)  # 向下移动一定是可用的
        
        # 测试放置数字玩家的可用动作
        self.game.is_player_turn = False
        moves = self.game.get_available_moves()
        empty_cells = self.game.get_empty_cells()
        self.assertEqual(len(moves), len(empty_cells) * 2)  # 每个空位置可以放2或4
        for move in moves:
            self.assertTrue(isinstance(move, tuple))
            self.assertTrue(move[1] in [2, 4])
            self.assertTrue(move[0] in empty_cells)
    
    def test_move_down(self):
        # 测试向下移动的情况
        test_cases = [
            {
                'initial': np.array([
                    [0, 0, 0, 2],
                    [0, 2, 4, 2],
                    [0, 2, 4, 8],
                    [0, 0, 4, 8]
                ]),
                'expected': np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 4, 4],
                    [0, 4, 8, 16]
                ]),
                'expected_score': 32
            },
            {
                'initial': np.array([
                    [2, 0, 0, 2],
                    [2, 0, 0, 2],
                    [2, 0, 0, 2],
                    [2, 0, 0, 2]
                ]),
                'expected': np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [4, 0, 0, 4],
                    [4, 0, 0, 4]
                ]),
                'expected_score': 16
            }
        ]
        
        for case in test_cases:
            self.game.board = case['initial'].copy()
            self.game.score = 0
            self.game.is_player_turn = True
            
            self.game.move(2)  # 向下移动
            
            np.testing.assert_array_equal(
                self.game.board, 
                case['expected'],
                err_msg=f"\nInitial:\n{case['initial']}\nExpected:\n{case['expected']}\nGot:\n{self.game.board}"
            )
            self.assertEqual(
                self.game.score,
                case['expected_score'],
                f"Score mismatch. Expected {case['expected_score']}, got {self.game.score}"
            )
            self.assertFalse(self.game.is_player_turn)  # 移动后应该切换到放置数字玩家
    
    def test_place_tile(self):
        """测试放置数字"""
        self.game.board = np.zeros((4, 4), dtype=int)
        self.game.is_player_turn = False
        
        # 测试放置2
        result = self.game.place_tile((0, 0), 2)
        self.assertTrue(result)
        self.assertEqual(self.game.board[0, 0], 2)
        self.assertTrue(self.game.is_player_turn)
        
        # 测试放置4
        self.game.is_player_turn = False
        result = self.game.place_tile((1, 1), 4)
        self.assertTrue(result)
        self.assertEqual(self.game.board[1, 1], 4)
        self.assertTrue(self.game.is_player_turn)
        
        # 测试在已有数字的位置放置
        self.game.is_player_turn = False
        result = self.game.place_tile((0, 0), 2)
        self.assertFalse(result)
        self.assertEqual(self.game.board[0, 0], 2)  # 值不应该改变
        self.assertFalse(self.game.is_player_turn)  # 玩家不应该切换
    
    def test_game_over(self):
        """测试游戏结束条件"""
        # 测试有空位时不应该结束
        self.game.board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 0, 4],
            [8, 16, 32, 64]
        ])
        self.assertFalse(self.game.is_game_over())
        
        # 测试有相邻相同数字时不应该结束
        self.game.board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 32]
        ])
        self.assertFalse(self.game.is_game_over())
        
        # 测试既无空位也无相邻相同数字时应该结束
        self.game.board = np.array([
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2, 4],
            [8, 16, 32, 64]
        ])
        self.assertTrue(self.game.is_game_over())

if __name__ == '__main__':
    unittest.main() 