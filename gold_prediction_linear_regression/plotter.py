import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_dataset(filename):
    """
    Reads a CSV file and returns the data as two numpy arrays: x (independent) and y (dependent).
    """
    data = pd.read_csv(filename, header=None, names=["x", "y"])
    x = data["x"].values
    y = data["y"].values
    return x, y

def plot_regression_line(x, y, m, b, title, color='blue'):
    """
    Plots the regression line and data points.

    Args:
        x: Independent variable.
        y: Dependent variable.
        m: Slope of the line.
        b: Intercept of the line.
        title: Title of the graph.
        color: Line color for the regression line.
    """
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of the data
    plt.scatter(x, y, label='Data Points', color='gray', alpha=0.6)
    
    # Regression line
    regression_y = m * x + b
    plt.plot(x, regression_y, label=f"y = {m:.2f}x + {b:.2f}", color=color, linewidth=2)
    
    # Graph formatting
    plt.title(title, fontsize=14)
    plt.xlabel("x (Independent Variable)", fontsize=12)
    plt.ylabel("y (Dependent Variable)", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()

# Read dataset
filename = "data.csv"
x, y = read_dataset(filename)

# Replace these with the actual slopes and intercepts from your C and Python implementations
m_c = 50  # Replace with the slope obtained from the C code
b_c = 1500  # Replace with the intercept obtained from the C code

m_python = 49.8  # Replace with the slope obtained from the Python code
b_python = 1502  # Replace with the intercept obtained from the Python code

# Plot for the C code
plot_regression_line(x, y, m_c, b_c, "Linear Regression (C Version)", color='red')

# Plot for the Python code
plot_regression_line(x, y, m_python, b_python, "Linear Regression (Python Version)", color='blue')
