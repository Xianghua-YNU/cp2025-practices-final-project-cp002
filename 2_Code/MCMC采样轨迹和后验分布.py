import numpy as np
np.random.seed(42)  # 放在初始化随机数前即可
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import emcee, corner
from celerite2 import terms, GaussianProcess

# === 读取数据 ===
t, y, yerr = np.loadtxt(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw.txt", unpack=True, skiprows=1)
sort_idx = np.argsort(t)
t, y, yerr = t[sort_idx], y[sort_idx], yerr[sort_idx]
if np.std(yerr) < 1e-3:
    yerr += 0.05

# === 模型构建 ===
def build_gp(params):
    log_tau, log_sigma = params
    tau, sigma = np.exp(log_tau), np.exp(log_sigma)
    kernel = terms.RealTerm(a=sigma**2, c=1/tau)
    return GaussianProcess(kernel)

def log_prior(params):
    log_tau, log_sigma = params
    if np.log(10) < log_tau < np.log(1000) and -5 < log_sigma < 1:
        return 0.0
    return -np.inf

def log_likelihood(params, t, y, yerr):
    try:
        gp = build_gp(params)
        gp.compute(t, yerr=yerr)
        return gp.log_likelihood(y)
    except:
        return -np.inf

def log_posterior(params, t, y, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(params, t, y, yerr)
    return lp + ll if np.isfinite(ll) else -np.inf

# === MCMC采样 ===
ndim, nwalkers = 2, 32
initial = np.log([132.6, 0.2])
pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True)
samples = sampler.get_chain(discard=1000, thin=10, flat=True)

# === 后验估计 & 角图 ===
log_tau_mcmc, log_sigma_mcmc = map(
    lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    zip(*np.percentile(samples, [16, 50, 84], axis=0))
)
print(f"log(tau) = {log_tau_mcmc[0]:.3f}, log(sigma) = {log_sigma_mcmc[0]:.3f}")

fig = corner.corner(samples,
    labels=[r"$\log(\tau)$（时间尺度）", r"$\log(\sigma)$（方差幅度）"],
    truths=[log_tau_mcmc[0], log_sigma_mcmc[0]],
    show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})
fig.suptitle("DRW模型参数的后验分布角图", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.2\相关性\优化后DRW角图.png", dpi=300)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import emcee, corner
from celerite2 import terms, GaussianProcess

# === 读取数据 ===
t, y, yerr = np.loadtxt(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw.txt", unpack=True, skiprows=1)
sort_idx = np.argsort(t)
t, y, yerr = t[sort_idx], y[sort_idx], yerr[sort_idx]
if np.std(yerr) < 1e-3:
    yerr += 0.05

# === 模型构建 ===
def build_gp(params):
    log_tau, log_sigma = params
    tau, sigma = np.exp(log_tau), np.exp(log_sigma)
    kernel = terms.RealTerm(a=sigma**2, c=1/tau)
    return GaussianProcess(kernel)

def log_prior(params):
    log_tau, log_sigma = params
    if np.log(10) < log_tau < np.log(1000) and -5 < log_sigma < 1:
        return 0.0
    return -np.inf

def log_likelihood(params, t, y, yerr):
    try:
        gp = build_gp(params)
        gp.compute(t, yerr=yerr)
        return gp.log_likelihood(y)
    except:
        return -np.inf

def log_posterior(params, t, y, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp): return -np.inf
    ll = log_likelihood(params, t, y, yerr)
    return lp + ll if np.isfinite(ll) else -np.inf

# === MCMC采样 ===
ndim, nwalkers = 2, 32
np.random.seed(42)  # 添加到 emcee 初始化前
initial = np.log([132.6, 0.2])
pos = initial + 1e-3 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True)
samples = sampler.get_chain(discard=1000, thin=10, flat=True)

# === 后验估计 & 角图 ===
log_tau_mcmc, log_sigma_mcmc = map(
    lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    zip(*np.percentile(samples, [16, 50, 84], axis=0))
)
print(f"log(tau) = {log_tau_mcmc[0]:.3f}, log(sigma) = {log_sigma_mcmc[0]:.3f}")

fig = corner.corner(samples,
    labels=[r"$\log(\tau)$（时间尺度）", r"$\log(\sigma)$（方差幅度）"],
    truths=[log_tau_mcmc[0], log_sigma_mcmc[0]],
    show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12})
fig.suptitle("DRW模型参数的后验分布角图", fontsize=14, y=1.0)
fig.subplots_adjust(top=0.88)
fig.tight_layout()
fig.savefig(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.2\相关性\DRW角图.png", dpi=300)



chain = sampler.get_chain()
nsteps, nwalkers, ndim = chain.shape
param_names = [r"$\log(\tau)$", r"$\log(\sigma)$"]

# 计算参数中位数
median_params = np.median(chain.reshape(-1, ndim), axis=0)

fig, axes = plt.subplots(ndim, 1, figsize=(12, 6), sharex=True)
for i in range(ndim):
    ax = axes[i]
    for w in range(nwalkers):
        ax.plot(chain[:, w, i], alpha=0.3)
    ax.set_ylabel(param_names[i], fontsize=14)
    ax.grid(True)

axes[-1].set_xlabel("采样步数 (step)", fontsize=14)

# 构造标题文本，包含参数估计值
title_text = (
    "MCMC采样过程轨迹图 (Trace Plot)\n"
    f"中位数估计: "
    rf"$\log(\tau)$ = {median_params[0]:.3f}, "
    rf"$\log(\sigma)$ = {median_params[1]:.3f}"
)

fig.suptitle(title_text, fontsize=16, y=1.0)  # y>1让标题稍微往上移一点
fig.subplots_adjust(top=0.88)  # 预留上方空间，防止标题被裁剪
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.2\相关性\MCMC采样轨迹图.png", dpi=300)
