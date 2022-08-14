import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2016-11-01'
end = '2021-10-31'

st.title('Stock Forecasting')

user_input = st.text_input('Enter Stock Symbol', 'TLKM.JK')
dataset = data.DataReader(user_input, 'yahoo', start, end)

#Describe Data
st.subheader('Data from 2016 - 2021')
st.write(dataset)

#Visualization
st.subheader('Close Price')
fig = plt.figure(figsize = (12,6))
plt.plot(dataset.Close)
st.pyplot(fig)

#Split Data
df = dataset.loc[:,['Close']]
# Partition data into data train, val & test
totaldata = dataset
totaldatatrain = int(len(totaldata)*0.6)
totaldataval = int(len(totaldata)*0.1)
totaldatatest = int(len(totaldata)*0.3)
# Store data into each partition
df_train= df[0:totaldatatrain]
df_valid=df[totaldatatrain:totaldatatrain+totaldataval]
df_test= df[totaldatatrain+totaldataval:]

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
df_train = sc.fit_transform(df_train)
df_valid = sc.fit_transform(df_valid)
df_test = sc.fit_transform(df_test)

model = load_model('gru_model.h5')

# Struktur Data Test Univariate
timestep = 15
x_test = []
y_test = []
for i in range(timestep, len(df_test)): 
    x_test.append(df_test[(i-timestep):i]) 
    y_test.append(df_test[i,0]) 
x_test, y_test = np.array(x_test), np.array(y_test) #Auto 3D
y_test = np.reshape(y_test, (y_test.shape[0], 1)) #supaya bisa di invers harus pny 2 dimensi
y_test = sc.inverse_transform(y_test)

y_predicted = model.predict(x_test)
y_predicted = sc.inverse_transform(y_predicted)
#sc = sc.scale_
#sc_factor = 1/sc[0]
#y_predicted = y_predicted * sc_factor
#y_test = y_test *sc_factor


#Prediction Plot 
st.subheader('Predictions vs Actual')
prediction_fig = plt.figure(figsize=(12,6))
plt.plot(y_test, color = 'red', label = 'Actual Price')
plt.plot(y_predicted, color = 'mediumblue', label = 'Predicted Price')
plt.xlabel('Period')
plt.ylabel('Price')
plt.legend()
st.pyplot(prediction_fig)

#Forecast Result 
df_pred = pd.DataFrame(y_predicted)
df_last = df_pred[-15:].values
df_last = sc.fit_transform(df_last)

x_forecast = []

forecast = df_last[-timestep:]
current_batch = forecast.reshape((1, timestep, 1))

n_forecast=90

for i in range(n_forecast):
  current_pred = model.predict(current_batch)[0]
  x_forecast.append(current_pred)
  current_batch = np.append(current_batch[:,1:,:],[[current_pred]], axis=1)
  true_predictions = sc.inverse_transform(x_forecast)

data=pd.DataFrame(df)
forecast_result=pd.DataFrame(data=true_predictions, columns=['Forecast'])
result = data.append(forecast_result,ignore_index=True)


st.subheader('Forecasting Result')
forecast_fig = plt.figure(figsize = (12,6))
plt.plot(result)
plt.legend(['Price', 'Forecast'])
st.pyplot(forecast_fig)
