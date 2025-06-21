import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合保存图片
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 指定中文字体：黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号 '-' 显示为方块的问题
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet, inv


import seaborn as sns
import corner
from celerite2 import terms, GaussianProcess

# 加载数据
t, y, yerr = np.loadtxt(
    r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw.txt",
    unpack=True, skiprows=1)

# 加载 MCMC 样本
samples = np.load(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.2\相关性\mcmc_samples(真实数据).npy")

# 取后验中位数参数
log_tau_best, log_sigma_best = np.median(samples, axis=0)
tau_best = np.exp(log_tau_best)
sigma_best = np.exp(log_sigma_best)

# 对时间序列排序，避免 ValueError: The input coordinates must be sorted
sorted_idx = np.argsort(t)
t = t[sorted_idx]
y = y[sorted_idx]
yerr = yerr[sorted_idx]

# 构建最优 GP
kernel = terms.RealTerm(a=sigma_best**2, c=1.0 / tau_best)
gp = GaussianProcess(kernel)
gp.compute(t, diag=yerr**2)



# ---------- 图 2：协方差函数拟合 ----------
def cov_func(t1, t2, sigma, tau):
    dt = np.abs(np.subtract.outer(t1, t2))
    return sigma**2 * np.exp(-dt / tau)

cov_matrix = cov_func(t_pred, t_pred, sigma_best, tau_best)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cov_matrix,
    cmap="magma",  # 更现代的色图
    xticklabels=False,
    yticklabels=False,
    cbar_kws={"label": "协方差强度"}
)

# 添加标题和文字说明
plt.title("DRW模型协方差函数可视化", fontsize=14)
plt.text(30, 50,
         r"协方差函数：$k(t_1, t_2) = \sigma^2 \, e^{-|t_1 - t_2|/\tau}$",
         fontsize=10, color='white',
         bbox=dict(facecolor='blue', alpha=0.5))


plt.tight_layout()
plt.savefig(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.4\协方差热力图(真实数据).png", dpi=300)
plt.close()


