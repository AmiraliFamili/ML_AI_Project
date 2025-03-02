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

# 5. Evaluation Functions
def evaluate_predictions(y_true, y_pred_samples):
    """Evaluate predictions from a Bayesian model with uncertainty"""
    # Calculate mean prediction for each test point
    y_pred_mean = np.mean(y_pred_samples, axis=0)

    # Calculate MSE and R²
    mse = mean_squared_error(y_true, y_pred_mean)
    r2 = r2_score(y_true, y_pred_mean)

    # Calculate credible intervals (90%)
    y_pred_5 = np.percentile(y_pred_samples, 5, axis=0)
    y_pred_95 = np.percentile(y_pred_samples, 95, axis=0)

    # Calculate interval width (uncertainty)
    interval_width = y_pred_95 - y_pred_5

    # Calculate coverage (percentage of true values within the 90% credible interval)
    coverage = np.mean((y_true >= y_pred_5) & (y_true <= y_pred_95))

    return {
        'mse': mse,
        'r2': r2,
        'mean_interval_width': np.mean(interval_width),
        'coverage': coverage,
        'predictions': {
            'mean': y_pred_mean,
            'lower': y_pred_5,
            'upper': y_pred_95
        }
    }
