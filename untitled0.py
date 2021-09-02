from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import keyboard


y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mse = mean_squared_error(y_true, y_pred)

print(mse)

plt.plot(y_true,'r',)
plt.plot(y_pred,'bo')

print ('\nholi \njejeje')