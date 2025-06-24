import numpy as np
import emcee
from celerite2 import terms, GaussianProcess

# 读取数据
t, y, yerr = np.loadtxt(
    r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.1\drw.txt",
    unpack=True, skiprows=1)

# 构建 log likelihood 函数
def log_likelihood(params):
    log_tau, log_sigma = params
    if not (0 < np.exp(log_tau) < 1000 and 0 < np.exp(log_sigma) < 100):
        return -np.inf
    try:
        kernel = terms.RealTerm(a=np.exp(2 * log_sigma), c=1 / np.exp(log_tau))
        gp = GaussianProcess(kernel)
        gp.compute(t, diag=yerr**2)
        return gp.log_likelihood(y)
    except Exception:
        return -np.inf

def log_prior(params):
    log_tau, log_sigma = params
    if -5 < log_tau < 10 and -10 < log_sigma < 5:
        return 0.0  # uniform prior
    return -np.inf

def log_posterior(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params)

# 初始化采样器
ndim = 2
nwalkers = 32
initial = np.array([np.log(100), np.log(1)])  # 合理初值
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
sampler.run_mcmc(pos, 3000, progress=True)

samples = sampler.get_chain(discard=500, flat=True)

# 保存样本
np.save(r"C:\Users\31025\OneDrive\桌面\计算物理\图片\4.2\相关性\mcmc_samples(真实数据).npy", samples)
