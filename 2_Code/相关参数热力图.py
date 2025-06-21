import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

from celerite2 import terms, GaussianProcess
import emcee
import corner
import seaborn as sns

# === 读取真实数据 ===
data_path = r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw.txt"
t, y, yerr = np.loadtxt(data_path, unpack=True, skiprows=1)

# 排序（GP要求时间升序）
sorted_idx = np.argsort(t)
t, y, yerr = t[sorted_idx], y[sorted_idx], yerr[sorted_idx]

# 为防止过拟合，若误差太小，则人为加一点
if np.std(yerr) < 1e-3:
    yerr += 0.05

# === 构建 DRW 高斯过程模型 ===
def build_gp(params):
    log_tau, log_sigma = params
    tau = np.exp(log_tau)
    sigma = np.exp(log_sigma)
    kernel = terms.RealTerm(a=sigma**2, c=1/tau)
    return GaussianProcess(kernel)

# === 后验概率函数 ===
def log_prior(params):
    log_tau, log_sigma = params
    if -10 < log_tau < 10 and -10 < log_sigma < 10:
        return 0.0
    return -np.inf

def log_likelihood(params, t, y, yerr):
    try:
        gp = build_gp(params)
        gp.compute(t, yerr=yerr)
        return gp.log_likelihood(y)
    except Exception:
        return -np.inf

def log_posterior(params, t, y, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(params, t, y, yerr)
    return lp + ll if np.isfinite(ll) else -np.inf

# === MCMC 采样设置 ===
ndim, nwalkers = 2, 32
initial = np.array([np.log(400), np.log(0.2)])
pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)  # 改为较大的扰动

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(t, y, yerr))
sampler.run_mcmc(pos, 8000, progress=True)
samples = sampler.get_chain(discard=2000, thin=5, flat=True)

# === 后验参数估计 ===
log_tau_mcmc, log_sigma_mcmc = map(
    lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    zip(*np.percentile(samples, [16, 50, 84], axis=0))
)

tau_est = np.exp(log_tau_mcmc[0])
sigma_est = np.exp(log_sigma_mcmc[0])
print(f"tau = {tau_est:.3f} ± {0.5*(log_tau_mcmc[1]+log_tau_mcmc[2]):.3f}")
print(f"sigma = {sigma_est:.3f} ± {0.5*(log_sigma_mcmc[1]+log_sigma_mcmc[2]):.3f}")


t_pred = np.linspace(t.min(), t.max(), 1000)
gp = build_gp(np.median(samples, axis=0))
gp.compute(t, yerr=yerr)
mu, var = gp.predict(y, t_pred, return_var=True)
std = np.sqrt(var)


# === 图：参数相关性热力图 ===
corr_matrix = np.corrcoef(samples.T)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
    xticklabels=[r"$\log(\tau)$", r"$\log(\sigma)$"],
    yticklabels=[r"$\log(\tau)$", r"$\log(\sigma)$"],
    cbar_kws={"label": "相关系数"}, linewidths=0.5, linecolor='white'
)
plt.title("参数后验相关性热力图", fontsize=14)


plt.tight_layout()
plt.savefig(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.2\相关性\DRW参数热力图.png", dpi=300)
plt.close()
