package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"
)

// GameMemory 存储游戏记录
type GameMemory struct {
	States   [][4][4]int `json:"states"`
	Policies [][]float64 `json:"policies"`
	Values   []float64   `json:"values"`
	Score    int         `json:"score"`
	MaxTile  int         `json:"max_tile"`
}

func selfPlay(model Model, numSimulations int, cPuct float64, tileActionSize int) *GameMemory {
	game := NewGame2048()
	mcts := NewMCTS(model, numSimulations, cPuct, tileActionSize)
	memory := &GameMemory{}

	for !game.IsGameOver() {
		if game.IsPlayerTurn() {
			state := game.GetBoard()
			start := time.Now()
			action, probs := mcts.GetActionProbs(game, 1.0)
			fmt.Println("state", state, "action", action, "probs", probs, "time", time.Since(start))

			// 记录状态和策略
			memory.States = append(memory.States, state)
			memory.Policies = append(memory.Policies, probs)

			ok := game.Move(action)
			if !ok {
				log.Fatalf("Invalid move: %d", action)
			}
		} else {
			action, _ := mcts.GetActionProbs(game, 1.0)
			ok := game.PlaceTileID(action)
			if !ok {
				log.Fatalf("Invalid tile placement: %d", action)
			}
		}
	}

	// 记录最终分数
	finalScore := game.GetScore()
	normalizedScore := normalizeScore(finalScore)
	for range memory.States {
		memory.Values = append(memory.Values, normalizedScore)
	}

	memory.Score = finalScore
	memory.MaxTile = game.GetMaxTile()

	return memory
}

func main() {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("Recovered from panic: %v", r)
			os.Exit(1)
		}
	}()

	// 配置参数
	numGames := 1
	numSimulations := 1000000
	cPuct := 1.0
	tileActionSize := 2

	// 创建HTTP模型客户端
	// model := NewHTTPModel()
	model := NewRandomModel()

	// 创建保存游戏记录的目录
	timestamp := time.Now().Format("20060102_150405")
	saveDir := filepath.Join("self_play_data", timestamp)
	if err := os.MkdirAll(saveDir, 0755); err != nil {
		log.Fatalf("Failed to create directory: %v", err)
	}

	// 进行自我对弈
	for i := 0; i < numGames; i++ {
		log.Printf("Starting game %d/%d", i+1, numGames)
		memory := selfPlay(model, numSimulations, cPuct, tileActionSize)

		// 保存游戏记录
		filename := filepath.Join(saveDir, fmt.Sprintf("game_%d.json", i))
		jsonData, err := json.MarshalIndent(memory, "", "  ")
		if err != nil {
			log.Printf("Error marshaling game data: %v", err)
			continue
		}

		if err := os.WriteFile(filename, jsonData, 0644); err != nil {
			log.Printf("Error saving game data: %v", err)
			continue
		}

		log.Printf("Game %d completed. Score: %d, Max Tile: %d", i+1, memory.Score, memory.MaxTile)
	}
}
