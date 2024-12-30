import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime
import platform

def set_platform_fonts():
    """根据不同平台设置合适的字体"""
    system = platform.system()
    
    # 常用的中文字体列表
    if system == 'Windows':
        font_list = ['Microsoft YaHei', 'SimHei', 'SimSun']
    elif system == 'Darwin':  # macOS
        font_list = ['Arial Unicode MS', 'Heiti TC', 'STHeiti']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK']
    
    # 尝试设置字体
    for font in font_list:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            # 测试字体是否可用
            fig = plt.figure()
            plt.text(0.5, 0.5, '测试中文', fontsize=12)
            plt.close(fig)
            print(f"使用字体: {font}")
            break
        except Exception:
            continue
    else:
        print("警告：未能找到合适的中文字体，将使用系统默认字体")
    
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def parse_log(log_file):
    iterations = []
    avg_scores = []
    max_scores = []
    policy_losses = []
    value_losses = []
    
    # 尝试不同的编码格式
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
    
    for encoding in encodings:
        try:
            with open(log_file, 'r', encoding=encoding) as f:
                lines = f.readlines()
                break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"读取文件时发生错误: {str(e)}")
            return [], [], [], [], []
    else:
        print("无法使用任何编码格式读取文件")
        return [], [], [], [], []
        
    for line in lines:
        try:
            # 解析平均分数和最高分数
            score_match = re.search(r'Iteration (\d+): Avg score = ([\d.]+), Max score = ([\d.]+)', line)
            if score_match:
                iteration = int(score_match.group(1))
                avg_score = float(score_match.group(2))
                max_score = float(score_match.group(3))
                iterations.append(iteration)
                avg_scores.append(avg_score)
                max_scores.append(max_score)
                
            # 解析策略损失和价值损失
            loss_match = re.search(r'Policy loss = ([\d.]+), Value loss = ([\d.]+)', line)
            if loss_match:
                policy_loss = float(loss_match.group(1))
                value_loss = float(loss_match.group(2))
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
        except Exception as e:
            print(f"解析行时发生错误: {str(e)}")
            continue
    
    if not iterations:
        print("警告：未能解析到任何数据")
        return [], [], [], [], []
        
    return iterations, avg_scores, max_scores, policy_losses, value_losses

def plot_training_curves(iterations, avg_scores, max_scores, policy_losses, value_losses):
    if not iterations:
        print("没有数据可供绘图")
        return
        
    # 设置字体
    set_platform_fonts()
        
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制分数曲线
    ax1.plot(iterations, avg_scores, label='Average Score', color='blue')
    ax1.plot(iterations, max_scores, label='Max Score', color='red')
    ax1.set_title('Game Score Progress During Training')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Score')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制损失曲线
    ax2.plot(iterations, policy_losses, label='Policy Loss', color='green')
    ax2.plot(iterations, value_losses, label='Value Loss', color='purple')
    ax2.set_title('Loss Function Changes During Training')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Loss')
    ax2.grid(True)
    ax2.legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    try:
        # 保存图表
        plt.savefig('training_curves.svg', format='svg', dpi=300, bbox_inches='tight')
        print("图表已保存为 training_curves.svg")
    except Exception as e:
        print(f"保存图表时发生错误: {str(e)}")
    finally:
        plt.close()

def main():
    # 解析日志文件
    iterations, avg_scores, max_scores, policy_losses, value_losses = parse_log('training.log')
    
    if not iterations:
        return
    
    # 绘制训练曲线
    plot_training_curves(iterations, avg_scores, max_scores, policy_losses, value_losses)
    
    # 输出一些统计信息
    print("\n训练统计信息:")
    print(f"训练总迭代次数: {len(iterations)}")
    print(f"最终平均分数: {avg_scores[-1]:.2f}")
    print(f"历史最高分数: {max(max_scores)}")
    print(f"最终策略损失: {policy_losses[-1]:.4f}")
    print(f"最终价值损失: {value_losses[-1]:.4f}")

if __name__ == "__main__":
    main() 