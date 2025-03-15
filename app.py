import numpy as np
import pandas as pd
import yfinance as yf 
from keras.models import load_model
import streamlit as st 
import matplotlib.pyplot as plt
 
model = load_model("C:/Users/kevin/Desktop/spp/sppmodel.keras")

st.header('Stock Price Predictor')

stock=st.text_input('Enter Stock Symbol','INFY')
start='2004-01-01'
end='2024-12-31'

data = yf.download(stock,start,end)

if data.empty:
    st.error("Incorrect stock code. Please enter a valid stock symbol.")
    st.stop()
else:
    st.subheader('Stock Data')
    st.write(data)

data_train = pd.DataFrame(data.Close[0:int(len(data)*0.8)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.8):len(data)]) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_90_days = data_train.tail(90)
data_test = pd.concat([pas_90_days,data_test],ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('30 Day - Price VS Moving Average ')
ma_30_days = data.Close.rolling(30).mean()
fig1 = plt.figure(figsize=(12,8))
plt.plot(ma_30_days,'r')
plt.plot(data.Close,'g')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['30 Days MA','Original Price'])
plt.show()
st.pyplot(fig1)

st.subheader('90 Day - Price VS Moving Average ')
ma_90_days = data.Close.rolling(90).mean()
fig2 = plt.figure(figsize=(12,8))
plt.plot(ma_90_days,'r')
plt.plot(data.Close,'g')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['90 Days MA','Original Price'])
plt.show()
st.pyplot(fig2)

st.subheader('180 Day - Price VS Moving Average ')
ma_180_days = data.Close.rolling(180).mean()
fig3 = plt.figure(figsize=(12,8))
plt.plot(ma_180_days,'r')
plt.plot(data.Close,'g')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(['180 Days MA','Original Price'])
plt.show()
st.pyplot(fig3)

x = []
y = []

for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y=np.array(x),np.array(y)

predict = model.predict(x)

scale = 1/scaler.scale_

predict = predict * scale 
y=y*scale

st.subheader('Original Price VS Predicted Price')
fig4 = plt.figure(figsize=(12,8))
plt.plot(predict,'r')
plt.plot(y,'g')
plt.xlabel('Time (Days)')
plt.ylabel('Price')
plt.legend(['Predicted Price','Original Price'])
plt.show()
st.pyplot(fig4)

