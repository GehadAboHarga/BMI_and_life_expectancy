# prediction of death age upon bmi
# application of single dimension linear regression

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")
X = bmi_life_data[['BMI']]
y = bmi_life_data[['Life expectancy']]

model = LinearRegression()
model.fit(X,y)
y_pred = model.predict([[21.07931]])
print(y_pred)

