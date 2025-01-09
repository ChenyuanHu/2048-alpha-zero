package main

import (
	"math"
	"math/rand"
)

const (
	normalizeScoreFactor = 4000000.0
)

func normalizeScore(score int) float64 {
	return float64(score) / normalizeScoreFactor
}

// Model 定义了神经网络模型的接口
type Model interface {
	Predict(board [4][4]int) ([]float64, float64)
}

// MCTSNode represents a node in the Monte Carlo Tree Search
type MCTSNode struct {
	model            Model // 这里需要定义一个Model接口
	gameState        *Game2048
	availableMoves   []int
	parent           *MCTSNode
	parentAction     int
	children         map[int]*MCTSNode
	numberOfVisits   int
	valueSum         float64
	priorProbability float64
	untriedActions   []int
	tileActionSize   int
	policy           []float64
	value            float64
}

// NewMCTSNode creates a new MCTS node
func NewMCTSNode(gameState *Game2048, parent *MCTSNode, parentAction int, model Model, tileActionSize int) *MCTSNode {
	node := &MCTSNode{
		model:            model,
		gameState:        gameState,
		availableMoves:   gameState.GetAvailableMoves(),
		parent:           parent,
		parentAction:     parentAction,
		children:         make(map[int]*MCTSNode),
		numberOfVisits:   1,
		valueSum:         0,
		priorProbability: 0,
		tileActionSize:   tileActionSize,
	}
	node.untriedActions = make([]int, len(node.availableMoves))
	copy(node.untriedActions, node.availableMoves)

	if len(node.availableMoves) == 0 {
		// 终局状态直接使用实际分数
		node.value = normalizeScore(node.gameState.GetScore())
		return node
	}

	// 模型预测的策略和价值
	if node.gameState.IsPlayerTurn() {
		// 移动方向玩家
		policy, value := node.model.Predict(node.gameState.GetBoard())
		node.value = value
		node.policy = make([]float64, 4)
		sum := 0.0
		for _, move := range node.availableMoves {
			node.policy[move] = policy[move]
			sum += policy[move]
		}
		// 归一化
		if sum > 0 {
			for i := range node.policy {
				node.policy[i] /= sum
			}
		}
		return node
	}

	// 放置数字玩家
	node.value = parent.value
	node.policy = make([]float64, 32)
	// 设置概率：2的概率为0.9，4的概率为0.1
	for i := 0; i < 16; i++ {
		node.policy[i] = 0.9
		node.policy[i+16] = 0.1
	}

	// 如果可用动作太多，随机选择部分动作
	if len(node.availableMoves) > node.tileActionSize {
		// 随机选择tileActionSize个动作
		selected := make([]int, node.tileActionSize)
		perm := rand.Perm(len(node.availableMoves))
		for i := 0; i < node.tileActionSize; i++ {
			selected[i] = node.availableMoves[perm[i]]
		}
		node.availableMoves = selected
		node.untriedActions = make([]int, len(selected))
		copy(node.untriedActions, selected)
	}

	// 归一化概率
	sum := 0.0
	for _, move := range node.availableMoves {
		sum += node.policy[move]
	}
	if sum > 0 {
		for i := range node.policy {
			node.policy[i] /= sum
		}
	}

	return node
}

// SelectChild selects the best child node according to the UCB formula
func (n *MCTSNode) SelectChild(cPuct float64) (int, *MCTSNode) {
	var bestScore float64 = -math.MaxFloat64
	var bestAction int
	var bestNode *MCTSNode

	for action, child := range n.children {
		exploitation := child.GetValue()
		nParent := float64(n.numberOfVisits)
		nChild := float64(child.numberOfVisits)
		exploration := child.priorProbability * math.Sqrt(nParent) / (1 + nChild)
		exploration *= cPuct

		score := exploitation + exploration
		if score > bestScore {
			bestScore = score
			bestAction = action
			bestNode = child
		}
	}

	return bestAction, bestNode
}

// Expand expands the current node with the given action
func (n *MCTSNode) Expand(action int, gameState *Game2048, priorProbability float64) *MCTSNode {
	node := NewMCTSNode(gameState, n, action, n.model, n.tileActionSize)
	node.priorProbability = priorProbability

	// 从未尝试动作中移除当前动作
	for i, a := range n.untriedActions {
		if a == action {
			n.untriedActions = append(n.untriedActions[:i], n.untriedActions[i+1:]...)
			break
		}
	}

	n.children[action] = node
	return node
}

// Update updates the node statistics
func (n *MCTSNode) Update(value float64) {
	n.numberOfVisits++
	n.valueSum += value
}

// GetValue returns the average value of the node
func (n *MCTSNode) GetValue() float64 {
	return n.valueSum / float64(n.numberOfVisits)
}

// MCTS represents the Monte Carlo Tree Search algorithm
type MCTS struct {
	model          Model
	numSimulations int
	cPuct          float64
	tileActionSize int
	root           *MCTSNode
	node           *MCTSNode
}

// NewMCTS creates a new MCTS instance
func NewMCTS(model Model, numSimulations int, cPuct float64, tileActionSize int) *MCTS {
	return &MCTS{
		model:          model,
		numSimulations: numSimulations,
		cPuct:          cPuct,
		tileActionSize: tileActionSize,
	}
}

// GetActionProbs returns the action probabilities for the current state
func (m *MCTS) GetActionProbs(gameState *Game2048, temperature float64) (int, []float64) {
	// 如果是新游戏或者根节点不存在，创建新的根节点
	if m.root == nil {
		m.root = NewMCTSNode(gameState.Clone(), nil, -1, m.model, m.tileActionSize)
		m.node = m.root
	}

	// 运行模拟
	for i := 0; i < m.numSimulations; i++ {
		node := m.node

		// Selection
		for len(node.untriedActions) == 0 && len(node.children) > 0 {
			_, node = node.SelectChild(m.cPuct)
		}

		// Expansion
		if len(node.untriedActions) > 0 {
			// 根据策略网络的概率选择动作
			actionSize := 4
			if !node.gameState.IsPlayerTurn() {
				actionSize = 32
			}

			// 计算未尝试动作的概率
			untriedProbs := make([]float64, actionSize)
			sum := 0.0
			for _, action := range node.untriedActions {
				untriedProbs[action] = node.policy[action]
				sum += node.policy[action]
			}

			// 归一化概率
			if sum > 0 {
				for i := range untriedProbs {
					untriedProbs[i] /= sum
				}
			} else {
				for i := range untriedProbs {
					untriedProbs[i] = 1.0 / float64(actionSize)
				}
			}

			// 选择动作
			action := weightedRandomChoice(untriedProbs)

			// 执行动作并扩展节点
			nextState := node.gameState.Clone()
			if node.gameState.IsPlayerTurn() {
				nextState.Move(action)
			} else {
				nextState.PlaceTileID(action)
			}
			node = node.Expand(action, nextState, node.policy[action])
		}

		value := node.value

		// Backup
		for node != m.node {
			node.Update(value)
			node = node.parent
		}
	}

	// 计算访问次数的概率分布
	actionSize := 4
	if !gameState.IsPlayerTurn() {
		actionSize = 32
	}

	visits := make([]float64, actionSize)
	for action, child := range m.node.children {
		visits[action] = math.Pow(float64(child.numberOfVisits), 1/temperature)
	}

	// 归一化访问次数
	sum := 0.0
	for _, v := range visits {
		sum += v
	}
	probs := make([]float64, actionSize)
	if sum > 0 {
		for i, v := range visits {
			probs[i] = v / sum
		}
	}

	// 选择动作
	action := weightedRandomChoice(probs)
	m.node = m.node.children[action]

	// 清理不在路径上的节点
	m.node.parent = nil
	m.root = m.node

	return action, probs
}

// weightedRandomChoice 根据概率分布随机选择一个动作
func weightedRandomChoice(probs []float64) int {
	r := rand.Float64()
	sum := 0.0
	for i, p := range probs {
		sum += p
		if r < sum {
			return i
		}
	}
	return len(probs) - 1
}
