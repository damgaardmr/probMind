import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
from pyro import poutine
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from pyro.infer import MCMC, HMC, NUTS

true_coefs = torch.tensor([1., 2., 3.])
data = torch.randn(2000, 3)
dim = 3
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()

def model(data):
    coefs_mean = torch.zeros(dim)
    coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
    y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
    return y

hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
mcmc = MCMC(hmc_kernel, num_samples=500, warmup_steps=100)
lol = mcmc.run(data)
samples = mcmc.get_samples()#['beta']
#print(samples)
#print(samples.mean(0))
print(samples)

