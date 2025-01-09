package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// PredictRequest 预测请求结构
type PredictRequest struct {
	Board [][]int `json:"board"`
}

// PredictResponse 预测响应结构
type PredictResponse struct {
	Policy []float64 `json:"policy"`
	Value  float64   `json:"value"`
}

// HTTPModel 实现Model接口的HTTP客户端
type HTTPModel struct {
	serverURL string
	client    *http.Client
}

func NewHTTPModel(serverURL string) *HTTPModel {
	return &HTTPModel{
		serverURL: serverURL,
		client: &http.Client{
			Transport: &http.Transport{
				ForceAttemptHTTP2: true,
				MaxIdleConns:      2,
				IdleConnTimeout:   10 * time.Second,
			},
		},
	}
}

func (m *HTTPModel) Predict(board [4][4]int) ([]float64, float64) {
	// 转换board为二维切片
	boardSlice := make([][]int, 4)
	for i := range boardSlice {
		boardSlice[i] = make([]int, 4)
		for j := range boardSlice[i] {
			boardSlice[i][j] = board[i][j]
		}
	}

	// 准备请求数据
	reqData := PredictRequest{Board: boardSlice}
	jsonData, err := json.Marshal(reqData)
	if err != nil {
		log.Printf("Error marshaling request: %v", err)
		return make([]float64, 4), 0.0
	}

	// 发送HTTP请求
	resp, err := m.client.Post(m.serverURL+"/predict", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Printf("Error making request: %v", err)
		return make([]float64, 4), 0.0
	}
	defer resp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Error reading response: %v", err)
		return make([]float64, 4), 0.0
	}

	// 解析响应
	var predictResp PredictResponse
	if err := json.Unmarshal(body, &predictResp); err != nil {
		log.Printf("Error unmarshaling response: %v", err)
		return make([]float64, 4), 0.0
	}

	return predictResp.Policy, predictResp.Value
}

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
			fmt.Println("state", state)
			action, probs := mcts.GetActionProbs(game, 1.0)

			// 记录状态和策略
			memory.States = append(memory.States, state)
			memory.Policies = append(memory.Policies, probs)

			ok := game.Move(action)
			if !ok {
				log.Printf("Invalid move: %d", action)
				break
			}
		} else {
			action, _ := mcts.GetActionProbs(game, 1.0)
			ok := game.PlaceTileID(action)
			if !ok {
				log.Printf("Invalid tile placement: %d", action)
				break
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
	// 配置参数
	serverURL := "http://127.0.0.1:5000"
	numGames := 1
	numSimulations := 800
	cPuct := 1.0
	tileActionSize := 2

	// 创建HTTP模型客户端
	model := NewHTTPModel(serverURL)

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
