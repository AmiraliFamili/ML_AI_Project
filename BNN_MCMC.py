import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
from pyro.optim import Adam
import time

class BayesianNN_MCMC:
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def model(self, x, y=None):
        # Priors for weights and biases - corrected to use to_event
        # First layer
        w1_prior = dist.Normal(0., 1.).expand([self.input_dim, self.hidden_dim]).to_event(2)
        b1_prior = dist.Normal(0., 1.).expand([self.hidden_dim]).to_event(1)

        # Second layer
        w2_prior = dist.Normal(0., 1.).expand([self.hidden_dim, self.hidden_dim]).to_event(2)
        b2_prior = dist.Normal(0., 1.).expand([self.hidden_dim]).to_event(1)

        # Output layer
        w3_prior = dist.Normal(0., 1.).expand([self.hidden_dim, self.output_dim]).to_event(2)
        b3_prior = dist.Normal(0., 1.).expand([self.output_dim]).to_event(1)

        # Sample from priors
        w1 = pyro.sample("w1", w1_prior)
        b1 = pyro.sample("b1", b1_prior)
        w2 = pyro.sample("w2", w2_prior)
        b2 = pyro.sample("b2", b2_prior)
        w3 = pyro.sample("w3", w3_prior)
        b3 = pyro.sample("b3", b3_prior)

        # Forward pass
        x1 = F.relu(torch.matmul(x, w1) + b1)
        x2 = F.relu(torch.matmul(x1, w2) + b2)
        mu = torch.matmul(x2, w3) + b3

        # Likelihood
        sigma = pyro.sample("sigma", dist.Uniform(0.05, 0.5))
        with pyro.plate("data", x.shape[0]):
            # Observe data
            pyro.sample("obs", dist.Normal(mu.squeeze(), sigma), obs=y)

        return mu

    def train(self, X_train, y_train, num_samples=500, warmup_steps=100):
        start_time = time.time()

        # Setup NUTS sampler
        nuts_kernel = NUTS(self.model)

        # Run MCMC
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        mcmc.run(X_train, y_train)

        # Get posterior samples
        self.posterior_samples = mcmc.get_samples()

        training_time = time.time() - start_time
        print(f"MCMC sampling completed in {training_time:.2f} seconds")

        return self.posterior_samples, training_time

    def predict(self, x_test):
        # Use Predictive to get samples from the posterior predictive distribution
        predictive = Predictive(self.model, posterior_samples=self.posterior_samples)
        predictions = predictive(x_test)
        return predictions["obs"].detach().numpy()