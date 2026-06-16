# linear-regression-from-scratch
A vectorized implementation of Linear Regression using NumPy

# Linear Regression from Scratch using NumPy

A vectorized implementation of Simple Linear Regression built from scratch using Python and NumPy. This project demonstrates the underlying mathematics of Gradient Descent without relying on high-level machine learning libraries like Scikit-Learn.

## Mathematical Formulation

- **Hypothesis (Prediction):** $$\hat{y} = w \cdot x + b$$

- **Cost Function (Mean Squared Error):** $$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Gradients Calculation:** $$\frac{\partial L}{\partial w} = \frac{1}{n} \sum -2x(y - \hat{y})$$  
  $$\frac{\partial L}{\partial b} = \frac{1}{n} \sum -2(y - \hat{y})$$

## Project Structure

- `LinearRegression.py`: Contains the core mathematical class and gradient descent optimization loop.
- `main.py`: A demonstration script to train the model on sample data and execute inference.

## How to Run

1. Clone the repository or download the files.
2. Run the main script:
```bash
python main.py
