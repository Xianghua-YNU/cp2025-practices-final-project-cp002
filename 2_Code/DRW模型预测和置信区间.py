import numpy as np
import matplotlib
matplotlib.use('Agg')  # 用于保存图像，不弹窗
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文字体
plt.rcParams['axes.unicode_minus'] = False

from celerite2 import terms, GaussianProcess
import emcee
import corner
import seaborn as sns

# === 读取数据 ===
t, y, yerr = np.loadtxt(
    r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw.txt",
    unpack=True, skiprows=1)
# 保证 t 是升序排列，y、yerr 跟着一起排
sorted_indices = np.argsort(t)
t = t[sorted_indices]
y = y[sorted_indices]
yerr = yerr[sorted_indices]

# 如果 yerr 太小，可以人为加大避免过拟合
if np.std(yerr) < 1e-3:
    yerr += 0.05

# === 构建 DRW GP ===
def build_gp(params):
    log_tau, log_sigma = params
    tau = np.exp(log_tau)
    sigma = np.exp(log_sigma)
    kernel = terms.RealTerm(a=sigma**2, c=1/tau)
    gp = GaussianProcess(kernel)
    return gp

def log_likelihood(params, t, y, yerr):
    try:
        gp = build_gp(params)
        gp.compute(t, yerr=yerr)
        return gp.log_likelihood(y)
    except:
        return -np.inf

def log_prior(params):
    log_tau, log_sigma = params
    if -10 < log_tau < 10 and -10 < log_sigma < 10:
        return 0.0
    return -np.inf

def log_posterior(params, t, y, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, t, y, yerr)
    return lp + ll if np.isfinite(ll) else -np.inf

# === MCMC 采样 ===
ndim, nwalkers = 2, 32
initial = np.array([np.log(400), np.log(0.2)])
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, y, yerr))
sampler.run_mcmc(pos, 3000, progress=True)
samples = sampler.get_chain(discard=1000, thin=10, flat=True)

# === 后验参数估计 ===
log_tau_mcmc, log_sigma_mcmc = map(
    lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
    zip(*np.percentile(samples, [16, 50, 84], axis=0))
)

print(f"tau = {np.exp(log_tau_mcmc[0]):.3f} ± {0.5 * (log_tau_mcmc[1] + log_tau_mcmc[2]):.3f}")
print(f"sigma = {np.exp(log_sigma_mcmc[0]):.3f} ± {0.5 * (log_sigma_mcmc[1] + log_sigma_mcmc[2]):.3f}")

# === 生成预测时间序列 ===
t_pred = np.linspace(min(t), max(t), 1000)

# === 使用最大后验参数进行预测 ===
best_params = np.median(samples, axis=0)
gp = build_gp(best_params)
gp.compute(t, yerr=yerr)
mu, var = gp.predict(y, t_pred, return_var=True)
std = np.sqrt(var)

# === 绘图：预测曲线 + 置信区间 ===
plt.figure(figsize=(10, 5))
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.4, label="观测数据")
plt.plot(t_pred, mu, "r", label="DRW 高斯过程拟合")
plt.fill_between(t_pred, mu - std, mu + std, color="red", alpha=0.3, label="1σ置信区间")
plt.xlabel("时间", fontsize=12)
plt.ylabel("观测值", fontsize=12)
plt.title("DRW模型预测与置信区间", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.2\相关性\DRW拟合预测曲线.png", dpi=300)
plt.close()
print("DRW拟合图已保存。")
