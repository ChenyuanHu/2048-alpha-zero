package main

import (
	"fmt"
	"math/rand"
)

// Game2048 represents the 2048 game state
type Game2048 struct {
	board        [4][4]int
	score        int
	isPlayerTurn bool // true表示移动方向的玩家，false表示放置数字的玩家
	step         int
}

// NewGame2048 creates and initializes a new game
func NewGame2048() *Game2048 {
	game := &Game2048{
		board:        [4][4]int{},
		score:        0,
		isPlayerTurn: true,
		step:         0,
	}
	game.addNewTile()
	game.addNewTile()
	return game
}

// Clone creates a deep copy of the game
func (g *Game2048) Clone() *Game2048 {
	newGame := &Game2048{
		board:        [4][4]int{},
		score:        g.score,
		isPlayerTurn: g.isPlayerTurn,
		step:         g.step,
	}
	copy(newGame.board[:], g.board[:])
	return newGame
}

// addNewTile adds a new tile (2 or 4) to a random empty cell
func (g *Game2048) addNewTile() {
	emptyCells := g.getEmptyCells()
	if len(emptyCells) == 0 {
		return
	}

	cell := emptyCells[rand.Intn(len(emptyCells))]
	value := 2
	if rand.Float64() > 0.9 {
		value = 4
	}
	g.board[cell[0]][cell[1]] = value
}

// getEmptyCells returns a list of empty cell coordinates
func (g *Game2048) getEmptyCells() [][2]int {
	emptyCells := make([][2]int, 0, 16) // 预分配16个空间，避免growslice
	for i := range g.board {
		for j := range g.board[i] {
			if g.board[i][j] == 0 {
				emptyCells = append(emptyCells, [2]int{i, j})
			}
		}
	}
	return emptyCells
}

// Move performs a move in the specified direction
// direction: 0=up, 1=right, 2=down, 3=left
func (g *Game2048) Move(direction int) bool {
	if !g.isPlayerTurn {
		return false
	}

	var moved bool
	originalScore := g.score

	switch direction {
	case 0: // 上移
		moved = g.moveUp()
	case 1: // 右移
		moved = g.moveRight()
	case 2: // 下移
		moved = g.moveDown()
	case 3: // 左移
		moved = g.moveLeft()
	}

	if !moved {
		g.score = originalScore
		return false
	}

	g.isPlayerTurn = false
	g.step++
	return true
}

// moveLeft moves all tiles to the left and merges them
func (g *Game2048) moveLeft() bool {
	moved := false
	for i := 0; i < 4; i++ {
		// 压缩行，移除空格
		line := make([]int, 0, 4)
		for j := 0; j < 4; j++ {
			if g.board[i][j] != 0 {
				line = append(line, g.board[i][j])
			}
		}

		// 如果这一行没有数字，跳过
		if len(line) == 0 {
			continue
		}

		// 合并相同的数字
		for j := 0; j < len(line)-1; j++ {
			if line[j] == line[j+1] {
				line[j] *= 2
				g.score += line[j]
				// 将后面的数字前移
				copy(line[j+1:], line[j+2:])
				line = line[:len(line)-1]
				moved = true
			}
		}

		// 检查是否发生了移动
		for j := 0; j < 4; j++ {
			newValue := 0
			if j < len(line) {
				newValue = line[j]
			}
			if g.board[i][j] != newValue {
				moved = true
			}
			g.board[i][j] = newValue
		}
	}
	return moved
}

// moveRight moves all tiles to the right and merges them
func (g *Game2048) moveRight() bool {
	moved := false
	for i := 0; i < 4; i++ {
		// 压缩行，移除空格
		line := make([]int, 0, 4)
		for j := 3; j >= 0; j-- {
			if g.board[i][j] != 0 {
				line = append(line, g.board[i][j])
			}
		}

		// 如果这一行没有数字，跳过
		if len(line) == 0 {
			continue
		}

		// 合并相同的数字
		for j := 0; j < len(line)-1; j++ {
			if line[j] == line[j+1] {
				line[j] *= 2
				g.score += line[j]
				// 将后面的数字前移
				copy(line[j+1:], line[j+2:])
				line = line[:len(line)-1]
				moved = true
			}
		}

		// 检查是否发生了移动
		for j := 0; j < 4; j++ {
			newValue := 0
			if j < len(line) {
				newValue = line[j]
			}
			if g.board[i][3-j] != newValue {
				moved = true
			}
			g.board[i][3-j] = newValue
		}
	}
	return moved
}

// moveUp moves all tiles up and merges them
func (g *Game2048) moveUp() bool {
	moved := false
	for j := 0; j < 4; j++ {
		// 压缩列，移除空格
		line := make([]int, 0, 4)
		for i := 0; i < 4; i++ {
			if g.board[i][j] != 0 {
				line = append(line, g.board[i][j])
			}
		}

		// 如果这一列没有数字，跳过
		if len(line) == 0 {
			continue
		}

		// 合并相同的数字
		for i := 0; i < len(line)-1; i++ {
			if line[i] == line[i+1] {
				line[i] *= 2
				g.score += line[i]
				// 将后面的数字前移
				copy(line[i+1:], line[i+2:])
				line = line[:len(line)-1]
				moved = true
			}
		}

		// 检查是否发生了移动
		for i := 0; i < 4; i++ {
			newValue := 0
			if i < len(line) {
				newValue = line[i]
			}
			if g.board[i][j] != newValue {
				moved = true
			}
			g.board[i][j] = newValue
		}
	}
	return moved
}

// moveDown moves all tiles down and merges them
func (g *Game2048) moveDown() bool {
	moved := false
	for j := 0; j < 4; j++ {
		// 压缩列，移除空格
		line := make([]int, 0, 4)
		for i := 3; i >= 0; i-- {
			if g.board[i][j] != 0 {
				line = append(line, g.board[i][j])
			}
		}

		// 如果这一列没有数字，跳过
		if len(line) == 0 {
			continue
		}

		// 合并相同的数字
		for i := 0; i < len(line)-1; i++ {
			if line[i] == line[i+1] {
				line[i] *= 2
				g.score += line[i]
				// 将后面的数字前移
				copy(line[i+1:], line[i+2:])
				line = line[:len(line)-1]
				moved = true
			}
		}

		// 检查是否发生了移动
		for i := 0; i < 4; i++ {
			newValue := 0
			if i < len(line) {
				newValue = line[i]
			}
			if g.board[3-i][j] != newValue {
				moved = true
			}
			g.board[3-i][j] = newValue
		}
	}
	return moved
}

// IsGameOver checks if the game is over
func (g *Game2048) IsGameOver() bool {
	// Check for empty cells
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if g.board[i][j] == 0 {
				return false
			}
		}
	}

	// Check for possible merges
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if i < 3 && g.board[i][j] == g.board[i+1][j] {
				return false
			}
			if j < 3 && g.board[i][j] == g.board[i][j+1] {
				return false
			}
		}
	}
	return true
}

// GetAvailableMoves returns a list of valid moves
func (g *Game2048) GetAvailableMoves() []int {
	if g.isPlayerTurn {
		// 移动方向玩家的可用动作
		var moves []int
		for direction := 0; direction < 4; direction++ {
			gameCopy := g.Clone()
			if gameCopy.Move(direction) {
				moves = append(moves, direction)
			}
		}
		return moves
	} else {
		// 放置数字玩家的可用动作
		emptyCells := g.getEmptyCells()
		moves := make([]int, 0, len(emptyCells)*2) // 预分配空间，每个空格可以放2或4
		for _, cell := range emptyCells {
			id := cell[0]*4 + cell[1]
			moves = append(moves, id) // 放置2的动作
		}
		for _, cell := range emptyCells {
			id := cell[0]*4 + cell[1]
			moves = append(moves, id+16) // 放置4的动作
		}
		return moves
	}
}

// GetScore returns the current score
func (g *Game2048) GetScore() int {
	return g.score
}

// GetMaxTile returns the highest tile value on the board
func (g *Game2048) GetMaxTile() int {
	maxTile := 0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if g.board[i][j] > maxTile {
				maxTile = g.board[i][j]
			}
		}
	}
	return maxTile
}

// GetBoard returns the current board state as [4][4]int
func (g *Game2048) GetBoard() [4][4]int {
	var board [4][4]int
	copy(board[:], g.board[:])
	return board
}

// String returns a string representation of the board
func (g *Game2048) String() string {
	var result string
	for i := range g.board {
		for j := range g.board[i] {
			result += fmt.Sprintf("%4d ", g.board[i][j])
		}
		result += "\n"
	}
	return result
}

// PlaceTileID places a new tile at the specified position
// vid is the action ID, where vid % 16 determines the position and vid >= 16 determines if it's a 4
func (g *Game2048) PlaceTileID(vid int) bool {
	if g.isPlayerTurn {
		return false
	}

	id := vid % 16
	value := 2
	if vid >= 16 {
		value = 4
	}

	row, col := id/4, id%4
	if g.board[row][col] != 0 {
		return false
	}

	g.board[row][col] = value
	g.isPlayerTurn = true
	return true
}

// GetStep returns the current step number
func (g *Game2048) GetStep() int {
	return g.step
}

// IsPlayerTurn returns whether it's the moving player's turn
func (g *Game2048) IsPlayerTurn() bool {
	return g.isPlayerTurn
}
