package main

import (
	"testing"
)

func TestMCTSRandomPlay(t *testing.T) {
	// 创建游戏和MCTS实例
	game := NewGame2048()
	model := &RandomModel{}
	mcts := NewMCTS(model, 1000, 1.0, 2)

	maxTile := 0
	maxScore := 0
	steps := 0
	temperature := 1.0

	// 运行游戏直到结束
	for !game.IsGameOver() {
		action, _ := mcts.GetActionProbs(game, temperature)
		if game.IsPlayerTurn() {
			// 移动方向玩家的回合
			game.Move(action)
		} else {
			// 放置数字玩家的回合
			game.PlaceTileID(action)
		}

		// 更新最大方块和分数
		if tile := game.GetMaxTile(); tile > maxTile {
			maxTile = tile
		}
		if score := game.GetScore(); score > maxScore {
			maxScore = score
		}
		steps++

		// 降低温度参数
		if steps > 30 {
			temperature = 0.1
		}
	}

	t.Logf("Game Over! Steps: %d, Max Tile: %d, Score: %d", steps, maxTile, maxScore)

	// 验证游戏结果
	if maxTile < 1024 {
		t.Errorf("Expected max tile >= 1024, got %d", maxTile)
	}
	if maxScore < 20000 {
		t.Errorf("Expected score >= 20000, got %d", maxScore)
	}
}

func TestMCTSNodeExpansion(t *testing.T) {
	model := &RandomModel{}
	game := NewGame2048()
	node := NewMCTSNode(game, nil, -1, model, 2)

	// 测试初始节点状态
	if len(node.availableMoves) == 0 {
		t.Error("Initial node should have available moves")
	}
	if len(node.children) != 0 {
		t.Error("Initial node should have no children")
	}
	if node.numberOfVisits != 1 {
		t.Error("Initial node should have 1 visit")
	}

	// 测试扩展节点
	if len(node.untriedActions) == 0 {
		t.Error("Initial node should have untried actions")
	}
	action := node.untriedActions[0]
	nextState := game.Clone()
	if game.IsPlayerTurn() {
		nextState.Move(action)
	} else {
		nextState.PlaceTileID(action)
	}
	child := node.Expand(action, nextState, 0.25)

	// 验证扩展后的状态
	if len(node.children) != 1 {
		t.Error("Node should have 1 child after expansion")
	}
	if child.parent != node {
		t.Error("Child's parent should be the original node")
	}
	if child.parentAction != action {
		t.Error("Child's parent action should match the expansion action")
	}
}

func TestMCTSSelection(t *testing.T) {
	model := &RandomModel{}
	game := NewGame2048()
	node := NewMCTSNode(game, nil, -1, model, 2)

	// 创建一些子节点
	for i := 0; i < 3; i++ {
		action := node.untriedActions[0]
		nextState := game.Clone()
		if game.IsPlayerTurn() {
			nextState.Move(action)
		} else {
			nextState.PlaceTileID(action)
		}
		child := node.Expand(action, nextState, 0.25)

		// 模拟一些访问
		for j := 0; j < i+1; j++ {
			child.Update(float64(j) * 0.1)
		}
	}

	// 测试选择
	action, selectedNode := node.SelectChild(1.0)
	if selectedNode == nil {
		t.Error("SelectChild should return a valid node")
	}
	if action < 0 {
		t.Error("SelectChild should return a valid action")
	}
}

func TestMCTSBackup(t *testing.T) {
	model := &RandomModel{}
	game := NewGame2048()
	node := NewMCTSNode(game, nil, -1, model, 2)

	// 创建一个子节点
	action := node.untriedActions[0]
	nextState := game.Clone()
	if game.IsPlayerTurn() {
		nextState.Move(action)
	} else {
		nextState.PlaceTileID(action)
	}
	child := node.Expand(action, nextState, 0.25)

	// 测试更新
	initialVisits := node.numberOfVisits
	initialValue := node.valueSum
	child.Update(0.5)
	node.Update(0.5)

	if node.numberOfVisits != initialVisits+1 {
		t.Error("Node visits should increase by 1")
	}
	if node.valueSum != initialValue+0.5 {
		t.Error("Node value sum should increase by 0.5")
	}
}

func TestMCTSGamePlay(t *testing.T) {
	// 运行多次游戏以验证稳定性
	numGames := 3
	successCount := 0

	for i := 0; i < numGames; i++ {
		game := NewGame2048()
		model := &RandomModel{}
		mcts := NewMCTS(model, 1000, 1.0, 2)
		maxTile := 0
		steps := 0

		for !game.IsGameOver() {
			action, _ := mcts.GetActionProbs(game, 1.0)
			if game.IsPlayerTurn() {
				game.Move(action)
			} else {
				game.PlaceTileID(action)
			}

			if tile := game.GetMaxTile(); tile > maxTile {
				maxTile = tile
			}
			steps++
		}

		t.Logf("Game %d - Steps: %d, Max Tile: %d, Score: %d", i+1, steps, maxTile, game.GetScore())
		if maxTile >= 1024 {
			successCount++
		}
	}

	// 至少有一局游戏达到1024
	if successCount == 0 {
		t.Error("Expected at least one game to reach 1024")
	}
}
