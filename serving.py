from flask import Flask, request, jsonify
import torch
import numpy as np
from neural_network import AlphaZeroNet
from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio

app = Flask(__name__)
model = None

def init_model():
    global model
    device = torch.device('cpu')
    model = AlphaZeroNet().to(device)
    
    try:
        checkpoint = torch.load('checkpoints/checkpoint_latest.pt', weights_only=True, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        # 启用torch的JIT优化
        model = torch.jit.script(model)
        print("Model loaded and optimized successfully")
    except FileNotFoundError:
        print("No checkpoint found!")
    
    model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if not isinstance(board := request.get_json().get('board', None), list):
        return jsonify({'error': 'Invalid board format'}), 400
    
    try:
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board).reshape(1, 4, 4)
            policy, value = model(board_tensor)
            return jsonify({
                'policy': policy[0].tolist(),
                'value': float(value[0][0])
            })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_model()
    config = Config()
    config.bind = ["127.0.0.1:5000"]
    config.alpn_protocols = ["h2", "http/1.1"]
    asyncio.run(serve(app, config)) 