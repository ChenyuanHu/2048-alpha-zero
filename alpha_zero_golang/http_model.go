package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
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

func NewHTTPModel() *HTTPModel {
	serverURL := "http://127.0.0.1:5000"
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

func (m *HTTPModel) Predict(gameState *Game2048) ([]float64, float64) {
	board := gameState.GetBoard()
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
		log.Fatalf("Error marshaling request: %v", err)
	}

	// 发送HTTP请求
	resp, err := m.client.Post(m.serverURL+"/predict", "application/json", bytes.NewBuffer(jsonData))
	if err != nil {
		log.Fatalf("Error making request: %v", err)
	}
	defer resp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalf("Error reading response: %v", err)
	}

	// 解析响应
	var predictResp PredictResponse
	if err := json.Unmarshal(body, &predictResp); err != nil {
		log.Fatalf("Error unmarshaling response: %v, error: %v", body, err)
	}

	return predictResp.Policy, predictResp.Value
}
