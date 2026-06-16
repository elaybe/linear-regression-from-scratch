from LinearRegression import LinearRegression
import numpy as np

X_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([3.0, 5.0, 7.0, 9.0])

model = LinearRegression(alpha=0.01,epochs=500)
model.fit(X_train,y_train)