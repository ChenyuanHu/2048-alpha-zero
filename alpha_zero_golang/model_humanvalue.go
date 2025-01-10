package main

// HumanValueModel 实现Model接口的人类价值模型
type HumanValueModel struct{}

func NewHumanValueModel() *HumanValueModel {
	return &HumanValueModel{}
}

func (m *HumanValueModel) Predict(gameState *Game2048) ([]float64, float64) {
	// 尝试所有4个方向的移动
	maxValue := -1.0
	maxValueDir := -1
	for dir := 0; dir < 4; dir++ {
		cloneState := gameState.Clone()
		if cloneState.Move(dir) {
			value := m.Value(cloneState)
			if value > maxValue {
				maxValue = value
				maxValueDir = dir
			}
		}
	}

	// 创建策略数组
	policy := make([]float64, 4)
	if maxValueDir != -1 {
		policy[maxValueDir] = 1.0
	} else {
		// 如果没有合法移动,使用均匀分布
		for i := 0; i < 4; i++ {
			policy[i] = 0.25
		}
	}
	return policy, m.Value(gameState)
}

func (m *HumanValueModel) Value(gameState *Game2048) float64 {
	board := gameState.GetBoard()
	score := 0.0

	// 1. 空白格子数量评分 (0-0.3)
	emptyCount := 0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if board[i][j] == 0 {
				emptyCount++
			}
		}
	}
	score += float64(emptyCount) / 16.0 * 0.3

	// 2. 最大数在角落评分 (0-0.2)
	maxTile := 0
	maxI, maxJ := 0, 0
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if board[i][j] > maxTile {
				maxTile = board[i][j]
				maxI, maxJ = i, j
			}
		}
	}
	if (maxI == 0 || maxI == 3) && (maxJ == 0 || maxJ == 3) {
		score += 0.2
	}

	// 3. 单调性评分 (0-0.5)
	// 检查行的单调性
	rowMonotonicity := 0.0
	for i := 0; i < 4; i++ {
		increasing := true
		decreasing := true
		for j := 1; j < 4; j++ {
			if board[i][j] > board[i][j-1] {
				decreasing = false
			}
			if board[i][j] < board[i][j-1] {
				increasing = false
			}
		}
		if increasing || decreasing {
			rowMonotonicity += 1.0
		}
	}

	// 检查列的单调性
	colMonotonicity := 0.0
	for j := 0; j < 4; j++ {
		increasing := true
		decreasing := true
		for i := 1; i < 4; i++ {
			if board[i][j] > board[i-1][j] {
				decreasing = false
			}
			if board[i][j] < board[i-1][j] {
				increasing = false
			}
		}
		if increasing || decreasing {
			colMonotonicity += 1.0
		}
	}

	score += (rowMonotonicity + colMonotonicity) / 8.0 * 0.5

	return score
}
