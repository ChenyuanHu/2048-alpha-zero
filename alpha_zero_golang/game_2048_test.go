package main

import (
	"fmt"
	"reflect"
	"testing"
)

func TestPlayerTurns(t *testing.T) {
	game := NewGame2048()

	// 设置初始状态
	game.board = [4][4]int{
		{2, 0, 0, 0},
		{2, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	}
	game.isPlayerTurn = true

	// 测试移动方向玩家移动后，应该切换到放置数字玩家
	if !game.Move(2) { // 向下移动
		t.Error("Expected valid move, got invalid")
	}
	if game.isPlayerTurn {
		t.Error("Expected player turn to be false after move")
	}

	// 测试放置数字玩家放置后，应该切换到移动方向玩家
	if !game.PlaceTileID(1) { // 在(0,1)位置放置2
		t.Error("Expected valid placement")
	}
	if !game.isPlayerTurn {
		t.Error("Expected player turn to be true after placement")
	}

	// 测试无效的移动不应该切换玩家
	game.board = [4][4]int{
		{2, 2, 2, 2},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	}
	game.isPlayerTurn = true
	if game.Move(0) { // 向上移动，无效
		t.Error("Expected invalid move")
	}
	if !game.isPlayerTurn {
		t.Error("Player turn should not change after invalid move")
	}

	// 测试无效的放置也不应该切换玩家
	game.isPlayerTurn = false
	if game.PlaceTileID(0) { // 已经有数字的位置
		t.Error("Expected invalid placement")
	}
	if game.isPlayerTurn {
		t.Error("Player turn should not change after invalid placement")
	}
}

func TestAvailableMoves(t *testing.T) {
	game := NewGame2048()

	// 测试移动方向玩家的可用动作
	game.board = [4][4]int{
		{2, 0, 0, 0},
		{2, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
	}
	game.isPlayerTurn = true

	moves := game.GetAvailableMoves()
	fmt.Println(moves)
	hasDown := false
	for _, move := range moves {
		if move == 2 { // 向下移动一定是可用的
			hasDown = true
		}
		if move < 0 || move > 3 {
			t.Errorf("Invalid move value: %d", move)
		}
	}
	if !hasDown {
		t.Error("Down move should be available")
	}

	// 测试放置数字玩家的可用动作
	game.isPlayerTurn = false
	moves = game.GetAvailableMoves()
	emptyCells := game.getEmptyCells()
	expectedMoveCount := len(emptyCells) * 2 // 每个空位置可以放2或4
	if len(moves) != expectedMoveCount {
		t.Errorf("Expected %d moves, got %d", expectedMoveCount, len(moves))
	}
}

func TestMoveDown(t *testing.T) {
	testCases := []struct {
		initial       [4][4]int
		expected      [4][4]int
		expectedScore int
	}{
		{
			initial: [4][4]int{
				{0, 0, 0, 2},
				{0, 2, 4, 2},
				{0, 2, 4, 8},
				{0, 0, 4, 8},
			},
			expected: [4][4]int{
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{0, 0, 4, 4},
				{0, 4, 8, 16},
			},
			expectedScore: 32,
		},
		{
			initial: [4][4]int{
				{2, 0, 0, 2},
				{2, 0, 0, 2},
				{2, 0, 0, 2},
				{2, 0, 0, 2},
			},
			expected: [4][4]int{
				{0, 0, 0, 0},
				{0, 0, 0, 0},
				{4, 0, 0, 4},
				{4, 0, 0, 4},
			},
			expectedScore: 16,
		},
	}

	for i, tc := range testCases {
		game := NewGame2048()
		game.board = tc.initial
		game.score = 0
		game.isPlayerTurn = true

		game.Move(2) // 向下移动

		if !reflect.DeepEqual(game.board, tc.expected) {
			t.Errorf("Case %d: board mismatch\nExpected:\n%v\nGot:\n%v", i, tc.expected, game.board)
		}
		if game.score != tc.expectedScore {
			t.Errorf("Case %d: score mismatch. Expected %d, got %d", i, tc.expectedScore, game.score)
		}
		if game.isPlayerTurn {
			t.Errorf("Case %d: player turn should be false after move", i)
		}
	}
}

func TestMoveUp(t *testing.T) {
	testCases := []struct {
		initial       [4][4]int
		expected      [4][4]int
		expectedScore int
	}{
		{
			initial: [4][4]int{
				{0, 0, 4, 8},
				{0, 2, 4, 8},
				{0, 2, 4, 2},
				{0, 0, 0, 2},
			},
			expected: [4][4]int{
				{0, 4, 8, 16},
				{0, 0, 4, 4},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
			expectedScore: 32,
		},
		{
			initial: [4][4]int{
				{2, 0, 0, 2},
				{2, 0, 0, 2},
				{2, 0, 0, 2},
				{2, 0, 0, 2},
			},
			expected: [4][4]int{
				{4, 0, 0, 4},
				{4, 0, 0, 4},
				{0, 0, 0, 0},
				{0, 0, 0, 0},
			},
			expectedScore: 16,
		},
	}

	for i, tc := range testCases {
		game := NewGame2048()
		game.board = tc.initial
		game.score = 0
		game.isPlayerTurn = true

		game.Move(0) // 向上移动

		if !reflect.DeepEqual(game.board, tc.expected) {
			t.Errorf("Case %d: board mismatch\nExpected:\n%v\nGot:\n%v", i, tc.expected, game.board)
		}
		if game.score != tc.expectedScore {
			t.Errorf("Case %d: score mismatch. Expected %d, got %d", i, tc.expectedScore, game.score)
		}
		if game.isPlayerTurn {
			t.Errorf("Case %d: player turn should be false after move", i)
		}
	}
}

func TestMoveLeft(t *testing.T) {
	testCases := []struct {
		initial       [4][4]int
		expected      [4][4]int
		expectedScore int
	}{
		{
			initial: [4][4]int{
				{0, 0, 2, 2},
				{0, 2, 2, 4},
				{0, 2, 2, 2},
				{2, 2, 2, 2},
			},
			expected: [4][4]int{
				{4, 0, 0, 0},
				{4, 4, 0, 0},
				{4, 2, 0, 0},
				{4, 4, 0, 0},
			},
			expectedScore: 20,
		},
		{
			initial: [4][4]int{
				{2, 2, 4, 8},
				{0, 0, 0, 8},
				{0, 0, 0, 0},
				{2, 2, 2, 2},
			},
			expected: [4][4]int{
				{4, 4, 8, 0},
				{8, 0, 0, 0},
				{0, 0, 0, 0},
				{4, 4, 0, 0},
			},
			expectedScore: 12,
		},
	}

	for i, tc := range testCases {
		game := NewGame2048()
		game.board = tc.initial
		game.score = 0
		game.isPlayerTurn = true

		game.Move(3) // 向左移动

		if !reflect.DeepEqual(game.board, tc.expected) {
			t.Errorf("Case %d: board mismatch\nExpected:\n%v\nGot:\n%v", i, tc.expected, game.board)
		}
		if game.score != tc.expectedScore {
			t.Errorf("Case %d: score mismatch. Expected %d, got %d", i, tc.expectedScore, game.score)
		}
		if game.isPlayerTurn {
			t.Errorf("Case %d: player turn should be false after move", i)
		}
	}
}

func TestMoveRight(t *testing.T) {
	testCases := []struct {
		initial       [4][4]int
		expected      [4][4]int
		expectedScore int
	}{
		{
			initial: [4][4]int{
				{2, 2, 0, 0},
				{4, 2, 2, 0},
				{2, 2, 2, 0},
				{2, 2, 2, 2},
			},
			expected: [4][4]int{
				{0, 0, 0, 4},
				{0, 0, 4, 4},
				{0, 0, 2, 4},
				{0, 0, 4, 4},
			},
			expectedScore: 20,
		},
		{
			initial: [4][4]int{
				{8, 4, 2, 2},
				{8, 0, 0, 0},
				{0, 0, 0, 0},
				{2, 2, 2, 2},
			},
			expected: [4][4]int{
				{0, 8, 4, 4},
				{0, 0, 0, 8},
				{0, 0, 0, 0},
				{0, 0, 4, 4},
			},
			expectedScore: 12,
		},
	}

	for i, tc := range testCases {
		game := NewGame2048()
		game.board = tc.initial
		game.score = 0
		game.isPlayerTurn = true

		game.Move(1) // 向右移动

		if !reflect.DeepEqual(game.board, tc.expected) {
			t.Errorf("Case %d: board mismatch\nExpected:\n%v\nGot:\n%v", i, tc.expected, game.board)
		}
		if game.score != tc.expectedScore {
			t.Errorf("Case %d: score mismatch. Expected %d, got %d", i, tc.expectedScore, game.score)
		}
		if game.isPlayerTurn {
			t.Errorf("Case %d: player turn should be false after move", i)
		}
	}
}

func TestPlaceTile(t *testing.T) {
	game := NewGame2048()
	game.board = [4][4]int{}
	game.isPlayerTurn = false

	// 测试放置2
	if !game.PlaceTileID(0) { // 在(0,0)位置放置2
		t.Error("Expected valid placement")
	}
	if game.board[0][0] != 2 {
		t.Errorf("Expected 2 at (0,0), got %d", game.board[0][0])
	}
	if !game.isPlayerTurn {
		t.Error("Expected player turn to be true after placement")
	}

	// 测试放置4
	game.isPlayerTurn = false
	if !game.PlaceTileID(21) { // 在(1,1)位置放置4 (16+5)
		t.Error("Expected valid placement")
	}
	if game.board[1][1] != 4 {
		t.Errorf("Expected 4 at (1,1), got %d", game.board[1][1])
	}
	if !game.isPlayerTurn {
		t.Error("Expected player turn to be true after placement")
	}

	// 测试在已有数字的位置放置
	game.isPlayerTurn = false
	if game.PlaceTileID(0) { // 在已有数字的位置(0,0)放置
		t.Error("Expected invalid placement")
	}
	if game.board[0][0] != 2 { // 值不应该改变
		t.Errorf("Expected 2 at (0,0), got %d", game.board[0][0])
	}
	if game.isPlayerTurn { // 玩家不应该切换
		t.Error("Player turn should not change after invalid placement")
	}
}

func TestGameOver(t *testing.T) {
	game := NewGame2048()

	// 测试有空位时不应该结束
	game.board = [4][4]int{
		{2, 4, 8, 16},
		{32, 64, 128, 256},
		{512, 1024, 0, 4},
		{8, 16, 32, 64},
	}
	if game.IsGameOver() {
		t.Error("Game should not be over when empty cells exist")
	}

	// 测试有相邻相同数字时不应该结束
	game.board = [4][4]int{
		{2, 4, 8, 16},
		{32, 64, 128, 256},
		{512, 1024, 2, 4},
		{8, 16, 32, 32},
	}
	if game.IsGameOver() {
		t.Error("Game should not be over when mergeable tiles exist")
	}

	// 测试既无空位也无相邻相同数字时应该结束
	game.board = [4][4]int{
		{2, 4, 8, 16},
		{32, 64, 128, 256},
		{512, 1024, 2, 4},
		{8, 16, 32, 64},
	}
	if !game.IsGameOver() {
		t.Error("Game should be over when no moves are possible")
	}
}
