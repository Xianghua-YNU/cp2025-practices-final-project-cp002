import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合保存图片
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 指定中文字体：黑体
plt.rcParams['axes.unicode_minus'] = False     # 解决负号 '-' 显示为方块的问题
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import slogdet, inv
from scipy.optimize import fmin_l_bfgs_b


# 中文显示支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体字体
plt.rcParams['axes.unicode_minus'] = False

# === 1. 读取数据 ===
file_path = r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw.txt"

# 如果有中文列标题，跳过首行
data = np.loadtxt(file_path, skiprows=1)
t = data[:, 0]
y = data[:, 1]
error = data[:, 2]

# === 2. 绘图 ===
plt.figure(figsize=(10, 5))
plt.errorbar(t, y, yerr=error, fmt='.', markersize=2.5, alpha=0.6,
             ecolor='gray', capsize=1.5, label='观测星等数据')

plt.xlabel("时间（天）", fontsize=14)
plt.ylabel("星等", fontsize=14)
plt.title("真实 DRW 光变曲线", fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

# === 3. 保存图像 ===
save_fig_path = r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw_lightcurve.png"
plt.savefig(save_fig_path, dpi=300)
plt.close()

print(" 光变曲线图已保存到 4.1 文件夹：drw_lightcurve.png")
