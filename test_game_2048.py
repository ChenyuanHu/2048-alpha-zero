import unittest
import numpy as np
from game_2048 import Game2048

class TestGame2048(unittest.TestCase):
    def setUp(self):
        self.game = Game2048()
    
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
                'expected_score': 32  # 2+2=4, 8+8=16, 4+4=8, 2+2=4
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
                'expected_score': 16  # 2+2=4, 2+2=4, 2+2=4, 2+2=4
            },
            {
                'initial': np.array([
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                'expected': np.array([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [4, 4, 4, 4]
                ]),
                'expected_score': 16  # 2+2=4 四次
            }
        ]
        
        for case in test_cases:
            self.game.board = case['initial'].copy()
            self.game.score = 0
            # 禁用随机添加新方块，以便验证移动结果
            original_add_new_tile = self.game.add_new_tile
            self.game.add_new_tile = lambda: None
            
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
            
            # 恢复原始的add_new_tile函数
            self.game.add_new_tile = original_add_new_tile
    
    def test_move_up(self):
        # 测试向上移动的情况
        test_cases = [
            {
                'initial': np.array([
                    [0, 0, 0, 2],
                    [0, 0, 0, 2],
                    [0, 2, 4, 2],
                    [2, 2, 4, 2]
                ]),
                'expected': np.array([
                    [2, 4, 8, 4],
                    [0, 0, 0, 4],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ]),
                'expected_score': 20  # 2+2=4, 2+2=4, 4+4=8, 2+2=4
            }
        ]
        
        for case in test_cases:
            self.game.board = case['initial'].copy()
            self.game.score = 0
            original_add_new_tile = self.game.add_new_tile
            self.game.add_new_tile = lambda: None
            
            self.game.move(0)  # 向上移动
            
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
            
            self.game.add_new_tile = original_add_new_tile

    def test_move_right(self):
        # 测试向右移动的情况
        test_cases = [
            {
                'initial': np.array([
                    [2, 2, 2, 0],
                    [0, 2, 2, 4],
                    [4, 4, 2, 2],
                    [2, 0, 2, 2]
                ]),
                'expected': np.array([
                    [0, 0, 2, 4],
                    [0, 0, 4, 4],
                    [0, 0, 8, 4],
                    [0, 0, 2, 4]
                ]),
                'expected_score': 24  # 2+2=4, 2+2=4, 2+2=4, 4+4=8, 2+2=4
            },
            {
                'initial': np.array([
                    [2, 0, 2, 2],
                    [2, 0, 2, 2],
                    [0, 0, 0, 0],
                    [4, 0, 4, 4]
                ]),
                'expected': np.array([
                    [0, 0, 2, 4],
                    [0, 0, 2, 4],
                    [0, 0, 0, 0],
                    [0, 0, 4, 8]
                ]),
                'expected_score': 16  # 2+2=4, 2+2=4, 4+4=8
            }
        ]
        
        for case in test_cases:
            self.game.board = case['initial'].copy()
            self.game.score = 0
            original_add_new_tile = self.game.add_new_tile
            self.game.add_new_tile = lambda: None
            
            self.game.move(1)  # 向右移动
            
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
            
            self.game.add_new_tile = original_add_new_tile

    def test_move_left(self):
        # 测试向左移动的情况
        test_cases = [
            {
                'initial': np.array([
                    [0, 2, 2, 2],
                    [4, 2, 2, 0],
                    [2, 2, 4, 4],
                    [2, 2, 0, 2]
                ]),
                'expected': np.array([
                    [4, 2, 0, 0],
                    [4, 4, 0, 0],
                    [4, 8, 0, 0],
                    [4, 2, 0, 0]
                ]),
                'expected_score': 24  # 2+2=4, 2+2=4, 2+2=4, 4+4=8, 2+2=4
            },
            {
                'initial': np.array([
                    [2, 2, 0, 2],
                    [2, 2, 0, 2],
                    [0, 0, 0, 0],
                    [4, 4, 0, 4]
                ]),
                'expected': np.array([
                    [4, 2, 0, 0],
                    [4, 2, 0, 0],
                    [0, 0, 0, 0],
                    [8, 4, 0, 0]
                ]),
                'expected_score': 16  # 2+2=4, 2+2=4, 4+4=8
            }
        ]
        
        for case in test_cases:
            self.game.board = case['initial'].copy()
            self.game.score = 0
            original_add_new_tile = self.game.add_new_tile
            self.game.add_new_tile = lambda: None
            
            self.game.move(3)  # 向左移动
            
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
            
            self.game.add_new_tile = original_add_new_tile

if __name__ == '__main__':
    unittest.main() 