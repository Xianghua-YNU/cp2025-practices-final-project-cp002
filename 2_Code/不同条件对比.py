import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import matplotlib.pyplot as plt
from celerite2 import terms, GaussianProcess

# 读取原始模拟数据
t_all, y_all, yerr_all = np.loadtxt(
    r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\simulated_drw_lightcurve.txt",
    unpack=True, skiprows=1)

# 构建 GP 函数
def build_gp(log_tau, log_sigma):
    kernel = terms.RealTerm(a=np.exp(2 * log_sigma), c=np.exp(-log_tau))
    return GaussianProcess(kernel)

# 不同子样本（时间 & 采样率）设置
conditions = {
    "长时间 + 稠密采样": (t_all, y_all, yerr_all),
    "稀疏采样": (t_all[::10], y_all[::10], yerr_all[::10]),
    "短时间序列": (t_all[:100], y_all[:100], yerr_all[:100])
}

# 使用统一的后验参数进行预测
log_tau_best, log_sigma_best = np.log(400), np.log(0.2)

plt.figure(figsize=(10, 6))

for label, (t, y, yerr) in conditions.items():
    gp = build_gp(log_tau_best, log_sigma_best)
    gp.compute(t, diag=yerr**2)  # 
    mu, var = gp.predict(y, t_all, return_var=True)
    std = np.sqrt(var)

    plt.plot(t_all, mu, label=label)
    plt.fill_between(t_all, mu - std, mu + std, alpha=0.15)

# 原始数据
plt.errorbar(t_all, y_all, yerr=yerr_all, fmt=".k", alpha=0.3, label="观测数据", zorder=-1)

plt.xlabel("时间")
plt.ylabel("观测值")
plt.title("不同条件下 DRW 模型预测曲线对比", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.3\不同条件下GP拟合曲线对比.png", dpi=300)
plt.close()
