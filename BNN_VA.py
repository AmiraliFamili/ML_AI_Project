
import torch
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS, Predictive
from pyro.optim import Adam

import time

class BayesianNN_VA:
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def model(self, x, y=None):
        # Priors for weights and biases
        # First layer - corrected to use to_event(2) for matrices
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

    def guide(self, x, y=None):
        # Variational parameters for w1
        w1_mu = pyro.param("w1_mu", torch.randn(self.input_dim, self.hidden_dim))
        w1_sigma = pyro.param("w1_sigma", torch.ones(self.input_dim, self.hidden_dim),
                             constraint=dist.constraints.positive)
        # Sample w1 - corrected to use to_event(2) for matrices
        w1 = pyro.sample("w1", dist.Normal(w1_mu, w1_sigma).to_event(2))

        # Variational parameters for b1
        b1_mu = pyro.param("b1_mu", torch.randn(self.hidden_dim))
        b1_sigma = pyro.param("b1_sigma", torch.ones(self.hidden_dim),
                             constraint=dist.constraints.positive)
        # Sample b1
        b1 = pyro.sample("b1", dist.Normal(b1_mu, b1_sigma).to_event(1))

        # Variational parameters for w2
        w2_mu = pyro.param("w2_mu", torch.randn(self.hidden_dim, self.hidden_dim))
        w2_sigma = pyro.param("w2_sigma", torch.ones(self.hidden_dim, self.hidden_dim),
                             constraint=dist.constraints.positive)
        # Sample w2
        w2 = pyro.sample("w2", dist.Normal(w2_mu, w2_sigma).to_event(2))

        # Variational parameters for b2
        b2_mu = pyro.param("b2_mu", torch.randn(self.hidden_dim))
        b2_sigma = pyro.param("b2_sigma", torch.ones(self.hidden_dim),
                             constraint=dist.constraints.positive)
        # Sample b2
        b2 = pyro.sample("b2", dist.Normal(b2_mu, b2_sigma).to_event(1))

        # Variational parameters for w3
        w3_mu = pyro.param("w3_mu", torch.randn(self.hidden_dim, self.output_dim))
        w3_sigma = pyro.param("w3_sigma", torch.ones(self.hidden_dim, self.output_dim),
                             constraint=dist.constraints.positive)
        # Sample w3
        w3 = pyro.sample("w3", dist.Normal(w3_mu, w3_sigma).to_event(2))

        # Variational parameters for b3
        b3_mu = pyro.param("b3_mu", torch.randn(self.output_dim))
        b3_sigma = pyro.param("b3_sigma", torch.ones(self.output_dim),
                             constraint=dist.constraints.positive)
        # Sample b3
        b3 = pyro.sample("b3", dist.Normal(b3_mu, b3_sigma).to_event(1))

        # Variational parameter for observation noise
        sigma_loc = pyro.param("sigma_loc", torch.tensor(0.1),
                              constraint=dist.constraints.positive)
        sigma = pyro.sample("sigma", dist.Delta(sigma_loc))

        return sigma

    def train(self, X_train, y_train, num_iterations=1000):
        pyro.clear_param_store()

        # Setup the optimizer
        optimizer = Adam({"lr": 0.01})
        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())

        # Training loop
        start_time = time.time()
        losses = []

        for j in range(num_iterations):
            loss = svi.step(X_train, y_train)
            losses.append(loss)

            if j % 100 == 0:
                print(f"[Iteration {j}] Loss: {loss}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        return losses, training_time

    def predict(self, x_test, num_samples=1000):
        x_test = x_test.to("cpu") 
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples)
        predictions = predictive(x_test)
        #print(predictions.device)
        return predictions["obs"].cpu().detach().numpy()