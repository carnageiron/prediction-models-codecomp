import pandas as pd
import numpy as np

def read_dataset(filename):
    """
    Reads a CSV file with two columns and returns two numpy arrays: x (independent variable) and y (dependent variable).
    """
    try:
        data = pd.read_csv(filename, header=None, names=['x', 'y'])
        x = data['x'].values
        y = data['y'].values
        return x, y
    except Exception as e:
        print(f"Error reading the dataset: {e}")
        return None, None

def normalize(data):
    """
    Normalizes the data to the range [0, 1].
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def linear_regression(x, y, learning_rate=0.01, iterations=1000):
    """
    Performs linear regression using gradient descent.

    Args:
        x: Independent variable.
        y: Dependent variable.
        learning_rate: Step size for gradient descent.
        iterations: Number of iterations to run gradient descent.

    Returns:
        m: Slope of the line.
        b: Intercept of the line.
    """
    m = 0.0  # Initial slope
    b = 0.0  # Initial intercept
    n = len(x)

    for i in range(iterations):
        # Predictions and errors
        predictions = m * x + b
        errors = predictions - y

        # Gradients
        dm = (2 / n) * np.dot(errors, x)
        db = (2 / n) * np.sum(errors)

        # Update parameters
        m -= learning_rate * dm
        b -= learning_rate * db

        # Compute loss and print for debugging
        loss = np.mean(errors ** 2)
        if i % 100 == 0:  # Print every 100 iterations
            print(f"Iteration {i}: Loss = {loss:.4f}, m = {m:.4f}, b = {b:.4f}")

        # Stop if gradients become invalid
        if np.isnan(dm) or np.isnan(db):
            print("NaN detected in gradients. Stopping gradient descent.")
            break

    return m, b

def predict(x, m, b):
    """
    Predicts the value of y given x, slope (m), and intercept (b).

    Args:
        x: Input value.
        m: Slope.
        b: Intercept.

    Returns:
        Predicted value of y.
    """
    return m * x + b

if __name__ == "__main__":
    # Read dataset
    filename = "data.csv"
    x, y = read_dataset(filename)
    if x is None or y is None:
        print("Failed to load dataset.")
        exit(1)

    # Normalize data
    x = normalize(x)
    y = normalize(y)

    # Train model
    learning_rate = 0.01
    iterations = 1000
    m, b = linear_regression(x, y, learning_rate, iterations)

    print(f"Trained Model: y = {m:.2f}x + {b:.2f}")

    # Predict for a new input
    try:
        user_input = float(input("Enter a value to predict: "))
        normalized_input = (user_input - np.min(x)) / (np.max(x) - np.min(x))
        prediction = predict(normalized_input, m, b)
        denormalized_prediction = prediction * (np.max(y) - np.min(y)) + np.min(y)
        print(f"Predicted value: {denormalized_prediction:.2f}")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
