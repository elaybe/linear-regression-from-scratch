import numpy as np

class LinearRegression:
    def __init__(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0


    def fit (self, x, y):
        for i in range(self.epochs):
            y_hat = self.w * x + self.b
            squared_errors = np.mean((y - y_hat) ** 2)

            dw = np.mean(-2 * (y - y_hat) * x)
            db = np.mean(-2 * (y - y_hat))

            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

        print(f"W= {self.w} \nB= {self.b}\nError= {squared_errors}")

