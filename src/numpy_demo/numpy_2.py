import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 查看系统可用字体
import matplotlib.font_manager
font_list = [f.name for f in matplotlib.font_manager.fontManager.ttflist]

# 选择支持中文的字体
plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # mac 上的字体
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子以保证结果可重复
np.random.seed(42)

# 生成正态分布数据
mu = 0      # 均值
sigma = 1   # 标准差
data = np.random.normal(mu, sigma, 10000)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制直方图
count, bins, ignored = plt.hist(data, bins=30, density=True,
                               alpha=0.6, color='blue',
                               label='随机样本直方图')

# 绘制理论上的正态分布曲线
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2,
         label='理论正态分布')

# 添加标题和标签
plt.title('正态分布展示 (μ=0, σ=1)', fontsize=14)
plt.xlabel('值', fontsize=12)
plt.ylabel('概率密度', fontsize=12)
plt.legend(fontsize=12)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)


# 显示图形
plt.show()
