// Package main provides the implementation of AlphaZero for 2048 game
package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"sync"
)

// Memory 存储游戏经验
type Memory struct {
	capacity int
	memory   []Experience
	mu       sync.RWMutex
}

// Experience 表示一个游戏经验
type Experience struct {
	State  [4][4]int
	Policy []float64
	Value  float64
}

// NewMemory 创建新的游戏记忆
func NewMemory(capacity int) *Memory {
	return &Memory{
		capacity: capacity,
		memory:   make([]Experience, 0, capacity),
	}
}

// Push 添加一个经验到记忆中
func (m *Memory) Push(state [4][4]int, policy []float64, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	exp := Experience{
		State:  state,
		Policy: make([]float64, len(policy)),
		Value:  value,
	}
	copy(exp.Policy, policy)

	if len(m.memory) >= m.capacity {
		// 移除最旧的经验
		m.memory = m.memory[1:]
	}
	m.memory = append(m.memory, exp)
}

// Sample 从记忆中采样
func (m *Memory) Sample(batchSize int) ([][4][4]int, [][]float64, []float64) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if batchSize > len(m.memory) {
		batchSize = len(m.memory)
	}

	// 生成随机索引
	indices := rand.Perm(len(m.memory))[:batchSize]

	states := make([][4][4]int, batchSize)
	policies := make([][]float64, batchSize)
	values := make([]float64, batchSize)

	for i, idx := range indices {
		states[i] = m.memory[idx].State
		policies[i] = make([]float64, len(m.memory[idx].Policy))
		copy(policies[i], m.memory[idx].Policy)
		values[i] = m.memory[idx].Value
	}

	return states, policies, values
}

// Len 返回记忆的当前大小
func (m *Memory) Len() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.memory)
}

// SaveToFile 将记忆保存到文件
func (m *Memory) SaveToFile(filename string) error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// 创建数据结构
	data := struct {
		States   [][4][4]int `json:"states"`
		Policies [][]float64 `json:"policies"`
		Values   []float64   `json:"values"`
	}{
		States:   make([][4][4]int, len(m.memory)),
		Policies: make([][]float64, len(m.memory)),
		Values:   make([]float64, len(m.memory)),
	}

	// 复制数据
	for i, exp := range m.memory {
		data.States[i] = exp.State
		data.Policies[i] = make([]float64, len(exp.Policy))
		copy(data.Policies[i], exp.Policy)
		data.Values[i] = exp.Value
	}

	// 保存为JSON文件
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("创建文件失败: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("编码数据失败: %v", err)
	}

	return nil
}
