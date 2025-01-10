package main

import (
	"math/rand"
	"time"
)

var rng = rand.New(rand.NewSource(time.Now().UnixNano()))

// RandomModel 实现Model接口的随机模型
type RandomModel struct{}

func NewRandomModel() *RandomModel {
	return &RandomModel{}
}

func (m *RandomModel) Predict(gameState *Game2048) ([]float64, float64) {
	// 返回随机的策略和价值
	policy := make([]float64, 32)
	for i := range policy {
		policy[i] = rng.Float64()
	}
	// 归一化策略
	sum := 0.0
	for _, p := range policy {
		sum += p
	}
	for i := range policy {
		policy[i] /= sum
	}
	// 返回随机价值
	value := rng.Float64()
	return policy, value
}
