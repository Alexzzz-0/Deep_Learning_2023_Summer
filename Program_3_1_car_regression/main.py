import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin','XX']
auto_mpg = pd.read_csv(url, names=column_names, delim_whitespace=True)

print("autompg size:", auto_mpg.shape)

# target = auto_mpg[:,0]
target = auto_mpg['MPG']
Cylinders_raw = auto_mpg['Cylinders']
print(Cylinders_raw.shape)
Cylinders = Cylinders_raw.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(Cylinders,target,test_size=0.2,random_state=8)

model = LinearRegression()
model.fit = (X_train, Y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test,Y_test,color='blue',label = 'Data')
plt.plot(X_test,y_pred,color='organe',label = 'label')
plt.legend()
plt.show()