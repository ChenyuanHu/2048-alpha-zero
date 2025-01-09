package main

import (
	"reflect"
	"testing"
)

func TestNewMemory(t *testing.T) {
	capacity := 100
	memory := NewMemory(capacity)

	if memory.capacity != capacity {
		t.Errorf("NewMemory capacity = %v, want %v", memory.capacity, capacity)
	}

	if len(memory.memory) != 0 {
		t.Errorf("NewMemory memory length = %v, want 0", len(memory.memory))
	}

	if cap(memory.memory) != capacity {
		t.Errorf("NewMemory memory capacity = %v, want %v", cap(memory.memory), capacity)
	}
}

func TestMemory_Push(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		pushNum  int
	}{
		{
			name:     "容量充足时的Push",
			capacity: 5,
			pushNum:  3,
		},
		{
			name:     "达到容量限制时的Push",
			capacity: 3,
			pushNum:  5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memory := NewMemory(tt.capacity)

			// 执行Push操作
			for i := 0; i < tt.pushNum; i++ {
				state := [4][4]int{{i, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}
				policy := []float64{float64(i), float64(i + 1)}
				value := float64(i)
				memory.Push(state, policy, value)
			}

			// 验证结果
			expectedLen := min(tt.pushNum, tt.capacity)
			if len(memory.memory) != expectedLen {
				t.Errorf("Memory length = %v, want %v", len(memory.memory), expectedLen)
			}

			// 验证最新的数据是否正确保存
			if len(memory.memory) > 0 {
				lastIdx := tt.pushNum - 1
				lastExp := memory.memory[len(memory.memory)-1]
				expectedState := [4][4]int{{lastIdx, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}
				expectedPolicy := []float64{float64(lastIdx), float64(lastIdx + 1)}
				expectedValue := float64(lastIdx)

				if !reflect.DeepEqual(lastExp.State, expectedState) {
					t.Errorf("Last state = %v, want %v", lastExp.State, expectedState)
				}
				if !reflect.DeepEqual(lastExp.Policy, expectedPolicy) {
					t.Errorf("Last policy = %v, want %v", lastExp.Policy, expectedPolicy)
				}
				if lastExp.Value != expectedValue {
					t.Errorf("Last value = %v, want %v", lastExp.Value, expectedValue)
				}
			}
		})
	}
}

func TestMemory_Sample(t *testing.T) {
	tests := []struct {
		name      string
		capacity  int
		pushNum   int
		batchSize int
	}{
		{
			name:      "记忆充足时的采样",
			capacity:  10,
			pushNum:   10,
			batchSize: 5,
		},
		{
			name:      "记忆不足时的采样",
			capacity:  5,
			pushNum:   3,
			batchSize: 5,
		},
		{
			name:      "空记忆的采样",
			capacity:  5,
			pushNum:   0,
			batchSize: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memory := NewMemory(tt.capacity)

			// 填充记忆
			for i := 0; i < tt.pushNum; i++ {
				state := [4][4]int{{i, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}
				policy := []float64{float64(i), float64(i + 1)}
				value := float64(i)
				memory.Push(state, policy, value)
			}

			// 执行采样
			states, policies, values := memory.Sample(tt.batchSize)

			// 验证采样结果
			expectedBatchSize := min(tt.batchSize, tt.pushNum)
			if len(states) != expectedBatchSize {
				t.Errorf("Sample states length = %v, want %v", len(states), expectedBatchSize)
			}
			if len(policies) != expectedBatchSize {
				t.Errorf("Sample policies length = %v, want %v", len(policies), expectedBatchSize)
			}
			if len(values) != expectedBatchSize {
				t.Errorf("Sample values length = %v, want %v", len(values), expectedBatchSize)
			}

			// 验证采样的数据格式是否正确
			for i := 0; i < expectedBatchSize; i++ {
				if len(policies[i]) != 2 {
					t.Errorf("Sample policy[%d] length = %v, want 2", i, len(policies[i]))
				}
			}
		})
	}
}

func TestMemory_Len(t *testing.T) {
	tests := []struct {
		name     string
		capacity int
		pushNum  int
		wantLen  int
	}{
		{
			name:     "空记忆",
			capacity: 5,
			pushNum:  0,
			wantLen:  0,
		},
		{
			name:     "部分填充",
			capacity: 5,
			pushNum:  3,
			wantLen:  3,
		},
		{
			name:     "完全填充",
			capacity: 5,
			pushNum:  5,
			wantLen:  5,
		},
		{
			name:     "超出容量",
			capacity: 5,
			pushNum:  7,
			wantLen:  5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			memory := NewMemory(tt.capacity)

			for i := 0; i < tt.pushNum; i++ {
				state := [4][4]int{{i, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}
				policy := []float64{float64(i), float64(i + 1)}
				value := float64(i)
				memory.Push(state, policy, value)
			}

			if got := memory.Len(); got != tt.wantLen {
				t.Errorf("Memory.Len() = %v, want %v", got, tt.wantLen)
			}
		})
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
