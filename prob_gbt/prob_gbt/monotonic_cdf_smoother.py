import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class SimpleMonotonicNN(nn.Module):
    def __init__(self, hidden_size=32):
        super(SimpleMonotonicNN, self).__init__()
        # First layer - can have any weights
        self.fc1 = nn.Linear(1, hidden_size)
        
        # Second layer - weights must be positive for monotonicity
        self.fc2_w = nn.Parameter(torch.rand(hidden_size, hidden_size) * 0.1)
        self.fc2_b = nn.Parameter(torch.zeros(hidden_size))
        
        # Output layer - weights must be positive for monotonicity
        self.fc3_w = nn.Parameter(torch.rand(hidden_size, 1) * 0.1)
        self.fc3_b = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.1)

    def forward(self, x):
        # First layer with sigmoid activation
        x = torch.sigmoid(self.fc1(x))
        
        # Second layer with positive weights and sigmoid activation
        x = torch.sigmoid(x @ torch.abs(self.fc2_w) + self.fc2_b)
        
        # Output layer with positive weights (no activation)
        x = x @ torch.abs(self.fc3_w) + self.fc3_b
        return x


def smooth_cdf(quantiles, y_pred_sample, hidden_size=32, num_epochs=5000, lr=0.01, 
               verbose=True, plot=True, patience=500, min_delta=1e-6):
    """
    Smooth a CDF using a monotonic neural network with early stopping.
    
    Parameters:
    -----------
    quantiles : numpy.ndarray
        The quantile values (y-axis of CDF)
    y_pred_sample : numpy.ndarray
        The predicted values (x-axis of CDF)
    hidden_size : int, optional
        Size of the hidden layers in the neural network
    num_epochs : int, optional
        Maximum number of training epochs
    lr : float, optional
        Learning rate for optimization
    verbose : bool, optional
        Whether to print progress during training
    plot : bool, optional
        Whether to generate and show plots
    patience : int, optional
        Number of epochs to wait for improvement before stopping
    min_delta : float, optional
        Minimum change in loss to qualify as an improvement
        
    Returns:
    --------
    model : SimpleMonotonicNN
        The trained neural network model
    y_smooth : numpy.ndarray
        The smoothed y values (x-axis of smoothed CDF)
    X_smooth_orig : numpy.ndarray
        The corresponding x values (y-axis of smoothed CDF)
    """
    # Normalize inputs
    quantiles_min, quantiles_max = quantiles.min(), quantiles.max()
    quantiles_norm = (quantiles - quantiles_min) / (quantiles_max - quantiles_min)

    y_pred_min, y_pred_max = y_pred_sample.min(), y_pred_sample.max()
    y_pred_norm = (y_pred_sample - y_pred_min) / (y_pred_max - y_pred_min)

    # Sort the data to ensure monotonicity in training
    sorted_indices = np.argsort(quantiles_norm)
    X_train = torch.tensor(quantiles_norm[sorted_indices], dtype=torch.float32).view(-1, 1)
    y_train = torch.tensor(y_pred_norm[sorted_indices], dtype=torch.float32).view(-1, 1)

    # Split data into training and validation sets (80/20 split)
    train_size = int(0.8 * len(X_train))
    X_train_split, X_val = X_train[:train_size], X_train[train_size:]
    y_train_split, y_val = y_train[:train_size], y_train[train_size:]

    # Initialize the model
    model = SimpleMonotonicNN(hidden_size=hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    # Training loop
    losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training step
        model.train()
        optimizer.zero_grad()
        y_pred_nn = model(X_train_split)
        
        train_loss = loss_fn(y_pred_nn, y_train_split)
        train_loss.backward()
        optimizer.step()
        
        losses.append(train_loss.item())
        
        # Validation step
        model.eval()
        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = loss_fn(y_pred_val, y_val)
            val_losses.append(val_loss.item())
        
        # Print progress
        if verbose and epoch % 500 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break
    
    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Generate smooth predictions
    X_smooth = torch.linspace(0, 1, 1000).view(-1, 1)
    with torch.no_grad():
        y_smooth = model(X_smooth).detach().numpy().flatten()

    # Scale back to original range
    y_smooth = y_smooth * (y_pred_max - y_pred_min) + y_pred_min
    X_smooth_orig = X_smooth.numpy().flatten() * (quantiles_max - quantiles_min) + quantiles_min

    # Plot if requested
    if plot:
        # Plot the smooth CDF
        plt.figure(figsize=(10, 6))
        plt.plot(y_pred_sample, quantiles, ".", label="Original CDF", alpha=0.7)
        plt.plot(y_smooth, X_smooth_orig, "-", label="Simple Monotonic NN", linewidth=2, color="r")
        plt.xlabel("Predicted Value")
        plt.ylabel("Quantile")
        plt.legend()
        plt.grid(True)
        plt.title("Smooth CDF using Simple Monotonic Neural Network")

        # Plot the loss curves
        plt.figure(figsize=(10, 4))
        plt.plot(losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)

        plt.show()

    return model, y_smooth, X_smooth_orig


def predict(model, x, x_min, x_max, y_min, y_max):
    """
    Make predictions with a trained monotonic neural network.
    
    Parameters:
    -----------
    model : SimpleMonotonicNN
        The trained neural network model
    x : numpy.ndarray
        The input values to predict on
    x_min, x_max : float
        The min and max values used for normalizing the training inputs
    y_min, y_max : float
        The min and max values used for scaling the training outputs
        
    Returns:
    --------
    y_pred : numpy.ndarray
        The predicted values
    """
    # Normalize inputs
    x_norm = (x - x_min) / (x_max - x_min)
    
    # Convert to tensor
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).view(-1, 1)
    
    # Make predictions
    with torch.no_grad():
        y_pred_norm = model(x_tensor).numpy().flatten()
    
    # Scale back to original range
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    
    return y_pred


# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(42)
    n_points = 100
    quantiles = np.linspace(0, 1, n_points)
    # Create a noisy sigmoid-like function
    y_pred_sample = 1 / (1 + np.exp(-10 * (quantiles - 0.5)))
    # Add some noise
    y_pred_sample += np.random.normal(0, 0.05, n_points)
    # Sort to ensure monotonicity
    y_pred_sample = np.sort(y_pred_sample)
    
    # Smooth the CDF with early stopping
    model, y_smooth, X_smooth = smooth_cdf(
        quantiles, 
        y_pred_sample, 
        hidden_size=32, 
        num_epochs=3000,
        lr=0.01,
        patience=500,  # Stop if no improvement for 500 epochs
        min_delta=1e-6  # Minimum improvement to reset patience counter
    ) 