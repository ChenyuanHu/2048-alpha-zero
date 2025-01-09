from flask import Flask, request, jsonify
import torch
import numpy as np
from neural_network import AlphaZeroNet
import os
import asyncio
from hypercorn.config import Config
from hypercorn.asyncio import serve

app = Flask(__name__)

# 全局变量存储模型
model = None
device = None

def load_latest_model():
    global model, device
    device_name = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    model = AlphaZeroNet().to(device)
    
    # 加载最新的检查点
    if os.path.exists('checkpoints/checkpoint_latest.pt'):
        checkpoint = torch.load('checkpoints/checkpoint_latest.pt', weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint and moved to {device}")
    else:
        print("No checkpoint found!")
    
    model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        board = data.get('board')
        
        if not board or not isinstance(board, list):
            return jsonify({'error': 'Invalid board format'}), 400
        
        # 转换输入格式
        board = np.array(board, dtype=np.float32).reshape(4, 4)
        
        # 使用模型进行预测
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board).unsqueeze(0).to(device)
            policy, value = model(board_tensor)
            
            # 转换为numpy数组并序列化
            policy = policy.cpu().numpy()[0].tolist()
            value = float(value.cpu().numpy()[0][0])
        
        return jsonify({
            'policy': policy,
            'value': value
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_latest_model()
    config = Config()
    config.bind = ["127.0.0.1:5000"]
    config.alpn_protocols = ["h2", "http/1.1"]
    asyncio.run(serve(app, config)) 