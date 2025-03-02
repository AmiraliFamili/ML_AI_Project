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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time

def plot_training_curve(losses, title="Training Loss"):
    """Plot the training curve for VA"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("va_training_curve.png")
    plt.close()

def plot_predictions(y_true, predictions_va, predictions_mcmc):
    """Plot predictions with uncertainty for both methods"""
    # Get a subset of test points for clearer visualization
    indices = np.arange(len(y_true))
    np.random.shuffle(indices)
    indices = indices[:30]  # Show 30 random test points

    plt.figure(figsize=(12, 8))

    # Sort indices by true value for better visualization
    sorted_indices = indices[np.argsort(y_true[indices])]

    # Extract predictions
    y_true_sorted = y_true[sorted_indices]

    # VA predictions
    y_va_mean = predictions_va['predictions']['mean'][sorted_indices]
    y_va_lower = predictions_va['predictions']['lower'][sorted_indices]
    y_va_upper = predictions_va['predictions']['upper'][sorted_indices]

    # MCMC predictions
    y_mcmc_mean = predictions_mcmc['predictions']['mean'][sorted_indices]
    y_mcmc_lower = predictions_mcmc['predictions']['lower'][sorted_indices]
    y_mcmc_upper = predictions_mcmc['predictions']['upper'][sorted_indices]

    x = np.arange(len(sorted_indices))

    # Plot VA
    plt.subplot(2, 1, 1)
    plt.plot(x, y_true_sorted, 'ko', label='True Values')
    plt.plot(x, y_va_mean, 'b-', label='VA Predictions')
    plt.fill_between(x, y_va_lower, y_va_upper, color='b', alpha=0.2, label='90% Credible Interval')
    plt.title('Variational Approximation Predictions')
    plt.ylabel('Wine Quality')
    plt.grid(True)
    plt.legend()

    # Plot MCMC
    plt.subplot(2, 1, 2)
    plt.plot(x, y_true_sorted, 'ko', label='True Values')
    plt.plot(x, y_mcmc_mean, 'r-', label='MCMC Predictions')
    plt.fill_between(x, y_mcmc_lower, y_mcmc_upper, color='r', alpha=0.2, label='90% Credible Interval')
    plt.title('MCMC Predictions')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Wine Quality')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("prediction_comparison.png")
    plt.close()

def plot_uncertainty_comparison(predictions_va, predictions_mcmc, y_test):
    """Compare uncertainty estimates between methods"""
    plt.figure(figsize=(12, 6))

    # VA uncertainties
    va_intervals = predictions_va['predictions']['upper'] - predictions_va['predictions']['lower']

    # MCMC uncertainties
    mcmc_intervals = predictions_mcmc['predictions']['upper'] - predictions_mcmc['predictions']['lower']

    # Absolute errors
    va_errors = np.abs(predictions_va['predictions']['mean'] - y_test)
    mcmc_errors = np.abs(predictions_mcmc['predictions']['mean'] - y_test)

    # Sort points by error magnitude for better visualization
    sorted_indices = np.argsort(va_errors)[:50]  # Focus on 50 points

    plt.subplot(1, 2, 1)
    plt.scatter(va_intervals[sorted_indices], va_errors[sorted_indices], alpha=0.7, label='VA')
    plt.scatter(mcmc_intervals[sorted_indices], mcmc_errors[sorted_indices], alpha=0.7, label='MCMC')
    plt.title('Uncertainty vs. Error')
    plt.xlabel('Prediction Interval Width')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot([va_intervals, mcmc_intervals], labels=['VA', 'MCMC'])
    plt.title('Uncertainty Distribution')
    plt.ylabel('Prediction Interval Width')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("uncertainty_comparison.png")
    plt.close()
