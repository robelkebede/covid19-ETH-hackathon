import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import plotly.offline as py

#date format(month,day,year)

data = pd.read_csv("./corona_dataset.csv")

m = Prophet()
m.fit(data)

future = m.make_future_dataframe(periods=1000)
forecast = m.predict(future)

pic = m.plot(forecast)


x = forecast["ds"]
y = forecast["yhat"]

plt.plot(x,y)
plt.show()
