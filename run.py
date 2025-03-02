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
from BNN import BayesianNeuralNetwork
from BNN_MCMC import BayesianNN_MCMC
from BNN_VA import BayesianNN_VA
from evaluate import evaluate_predictions
from plotting import *

# 6. Main Execution Function
def run_comparison():
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess_data()
    print(f"Dataset loaded: {len(X_train)} training examples, {len(X_test)} test examples")

    # Define model dimensions
    input_dim = X_train.shape[1]
    hidden_dim = 20  # Smaller network for faster computation

    # Create results dictionary
    results = {
        'VA': {},
        'MCMC': {}
    }

    # 1. Train and evaluate VA model
    print("\n" + "="*50)
    print("Training Variational Approximation model...")
    va_model = BayesianNN_VA(input_dim, hidden_dim)
    va_losses, va_time = va_model.train(X_train, y_train, num_iterations=1000)

    # Get VA predictions
    print("Making VA predictions...")
    va_predictions = va_model.predict(X_test)
    va_results = evaluate_predictions(y_test.numpy(), va_predictions)
    va_results['training_time'] = va_time
    results['VA'] = va_results

    # 2. Train and evaluate MCMC model (with smaller network and fewer samples for tractability)
    print("\n" + "="*50)
    print("Training MCMC model (this may take a while)...")
    mcmc_model = BayesianNN_MCMC(input_dim, 10)  # Smaller hidden dim for MCMC
    mcmc_samples, mcmc_time = mcmc_model.train(X_train, y_train, num_samples=200, warmup_steps=100)

    # Get MCMC predictions
    print("Making MCMC predictions...")
    mcmc_predictions = mcmc_model.predict(X_test)
    mcmc_results = evaluate_predictions(y_test.numpy(), mcmc_predictions)
    mcmc_results['training_time'] = mcmc_time
    results['MCMC'] = mcmc_results

    # 3. Generate comparison visualizations
    print("\n" + "="*50)
    print("Generating visualizations...")
    plot_training_curve(va_losses)
    plot_predictions(y_test.numpy(), va_results, mcmc_results)
    plot_uncertainty_comparison(va_results, mcmc_results, y_test.numpy())

    # 4. Print comparison results
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)

    print(f"VA Training Time: {va_results['training_time']:.2f} seconds")
    print(f"MCMC Training Time: {mcmc_results['training_time']:.2f} seconds")
    print(f"Time Ratio (MCMC/VA): {mcmc_results['training_time']/va_results['training_time']:.2f}x")

    print("\nPrediction Accuracy:")
    print(f"VA MSE: {va_results['mse']:.4f}, R²: {va_results['r2']:.4f}")
    print(f"MCMC MSE: {mcmc_results['mse']:.4f}, R²: {mcmc_results['r2']:.4f}")

    print("\nUncertainty Quantification:")
    print(f"VA Mean Interval Width: {va_results['mean_interval_width']:.4f}, Coverage: {va_results['coverage']:.2%}")
    print(f"MCMC Mean Interval Width: {mcmc_results['mean_interval_width']:.4f}, Coverage: {mcmc_results['coverage']:.2%}")

    return results