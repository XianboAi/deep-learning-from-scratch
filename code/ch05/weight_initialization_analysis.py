# 使用AI帮助分析为什么初始权重对于收敛速度影响这么大
import numpy as np
import matplotlib.pyplot as plt

def analyze_weight_initialization():
    """分析不同权重初始化对前向传播的影响"""
    
    # 模拟输入数据 (MNIST像素值归一化后)
    x = np.random.randn(100, 784) * 0.1  # 标准化的输入
    
    # test2方式：大权重
    W_bad = np.random.rand(784, 100)  # [0, 1)
    b_bad = np.zeros(100)
    
    # 5.7方式：小权重
    W_good = 0.01 * np.random.randn(784, 100)  # N(0, 0.01²)
    b_good = np.zeros(100)
    
    # 前向传播
    z_bad = np.dot(x, W_bad) + b_bad
    z_good = np.dot(x, W_good) + b_good
    
    # ReLU激活
    a_bad = np.maximum(0, z_bad)
    a_good = np.maximum(0, z_good)
    
    print("=== 权重初始化影响分析 ===")
    print(f"输入数据范围: [{x.min():.3f}, {x.max():.3f}]")
    print(f"输入标准差: {x.std():.3f}")
    print()
    
    print("--- test2方式 (大权重) ---")
    print(f"权重范围: [{W_bad.min():.3f}, {W_bad.max():.3f}]")
    print(f"权重均值: {W_bad.mean():.3f}, 标准差: {W_bad.std():.3f}")
    print(f"激活前范围: [{z_bad.min():.3f}, {z_bad.max():.3f}]")
    print(f"激活前标准差: {z_bad.std():.3f}")
    print(f"激活后范围: [{a_bad.min():.3f}, {a_bad.max():.3f}]")
    print(f"激活后标准差: {a_bad.std():.3f}")
    print(f"神经元死亡率 (输出为0的比例): {(a_bad == 0).mean():.3f}")
    print()
    
    print("--- 5.7方式 (小权重) ---")
    print(f"权重范围: [{W_good.min():.3f}, {W_good.max():.3f}]")
    print(f"权重均值: {W_good.mean():.3f}, 标准差: {W_good.std():.3f}")
    print(f"激活前范围: [{z_good.min():.3f}, {z_good.max():.3f}]")
    print(f"激活前标准差: {z_good.std():.3f}")
    print(f"激活后范围: [{a_good.min():.3f}, {a_good.max():.3f}]")
    print(f"激活后标准差: {a_good.std():.3f}")
    print(f"神经元死亡率 (输出为0的比例): {(a_good == 0).mean():.3f}")
    print()
    
    # 梯度分析
    print("=== 梯度传播分析 ===")
    
    # 模拟反向传播的梯度
    dout = np.ones_like(a_bad)  # 假设上层传来的梯度为1
    
    # ReLU反向传播
    dz_bad = dout.copy()
    dz_bad[a_bad == 0] = 0
    
    dz_good = dout.copy()
    dz_good[a_good == 0] = 0
    
    # 权重梯度
    dW_bad = np.dot(x.T, dz_bad)
    dW_good = np.dot(x.T, dz_good)
    
    print("--- test2方式梯度 ---")
    print(f"权重梯度范围: [{dW_bad.min():.6f}, {dW_bad.max():.6f}]")
    print(f"权重梯度标准差: {dW_bad.std():.6f}")
    print(f"权重梯度均值: {dW_bad.mean():.6f}")
    print()
    
    print("--- 5.7方式梯度 ---")
    print(f"权重梯度范围: [{dW_good.min():.6f}, {dW_good.max():.6f}]")
    print(f"权重梯度标准差: {dW_good.std():.6f}")
    print(f"权重梯度均值: {dW_good.mean():.6f}")
    print()
    
    print(f"梯度标准差比值: {dW_bad.std() / dW_good.std():.2f}")
    
    return {
        'bad': {'W': W_bad, 'z': z_bad, 'a': a_bad, 'dW': dW_bad},
        'good': {'W': W_good, 'z': z_good, 'a': a_good, 'dW': dW_good}
    }

def visualize_comparison(results):
    """可视化对比"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 权重分布
    axes[0, 0].hist(results['bad']['W'].flatten(), bins=50, alpha=0.7, label='test2 (大权重)', color='red')
    axes[0, 0].hist(results['good']['W'].flatten(), bins=50, alpha=0.7, label='5.7 (小权重)', color='blue')
    axes[0, 0].set_title('权重分布对比')
    axes[0, 0].legend()
    
    # 激活前分布
    axes[0, 1].hist(results['bad']['z'].flatten(), bins=50, alpha=0.7, label='test2', color='red')
    axes[0, 1].hist(results['good']['z'].flatten(), bins=50, alpha=0.7, label='5.7', color='blue')
    axes[0, 1].set_title('激活前分布对比')
    axes[0, 1].legend()
    
    # 激活后分布
    axes[0, 2].hist(results['bad']['a'].flatten(), bins=50, alpha=0.7, label='test2', color='red')
    axes[0, 2].hist(results['good']['a'].flatten(), bins=50, alpha=0.7, label='5.7', color='blue')
    axes[0, 2].set_title('激活后分布对比 (ReLU)')
    axes[0, 2].legend()
    
    # 梯度分布
    axes[1, 0].hist(results['bad']['dW'].flatten(), bins=50, alpha=0.7, label='test2', color='red')
    axes[1, 0].hist(results['good']['dW'].flatten(), bins=50, alpha=0.7, label='5.7', color='blue')
    axes[1, 0].set_title('权重梯度分布对比')
    axes[1, 0].legend()
    
    # 神经元激活状态
    bad_dead_ratio = (results['bad']['a'] == 0).mean()
    good_dead_ratio = (results['good']['a'] == 0).mean()
    
    axes[1, 1].bar(['test2', '5.7'], [bad_dead_ratio, good_dead_ratio], color=['red', 'blue'])
    axes[1, 1].set_title('神经元死亡率对比')
    axes[1, 1].set_ylabel('死亡率')
    
    # 梯度大小对比
    axes[1, 2].bar(['test2', '5.7'], 
                   [results['bad']['dW'].std(), results['good']['dW'].std()], 
                   color=['red', 'blue'])
    axes[1, 2].set_title('梯度标准差对比')
    axes[1, 2].set_ylabel('梯度标准差')
    
    plt.tight_layout()
    plt.savefig('weight_initialization_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    results = analyze_weight_initialization()
    
    try:
        visualize_comparison(results)
        print("可视化图表已保存为 'weight_initialization_comparison.png'")
    except ImportError:
        print("matplotlib未安装，跳过可视化")