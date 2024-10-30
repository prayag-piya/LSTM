from LSTM.Network import *
from scraper.yahoofinance import Scraper
import os 
import numpy as np
import pandas as pd
import locale
from locale import atof
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def dataFinder():
    scraper = Scraper("https://finance.yahoo.com/quote/NVDA/history/?period1=917015400&period2=1728398759", "NVDA")
    scraper.request()
    scraper.load_file()
    scraper.parser("NVDA.html")
    scraper.csvwriter()
    
def main():
    if not os.path.exists("NVDA.csv"):
        dataFinder()
    df = pd.read_csv("NVDA.csv")
    df.columns = ['Date', 'Open', 'High', "Low", "Close", "AddjClose", "Volume"]
    df = df[::-1]
    df = df.dropna()
    df["Open"] = df.Open.astype(float)

    locale.setlocale(locale.LC_NUMERIC, '')
    df["Volume"] = df["Volume"].map(atof)

    training_set = df.iloc[:, 2:4].values

    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)


    X_data = []
    y_data = []
    for i in range(60, len(training_set_scaled)):
        X_data.append(training_set_scaled[i-60:i, 0])
        y_data.append(training_set_scaled[i, 0])
    X_data, y_data = np.array(X_data), np.array(y_data)

    X_train1, X_test1, y_train1, y_test1 = train_test_split(
        X_data, y_data, test_size=0.2, shuffle=False
    )

    X_train = np.reshape(X_train1, (X_train1.shape[0], X_train1.shape[1], 1))
    X_test = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))

    lstm = LSTM(60, 100)
    T = max(X_train.flatten().shape)
    dense1 = DenseLayer(100, 50)
    dense2 = DenseLayer(50, 1)

    lr = 0.0001
    epoch = 100
    monitor = np.zeros((100))

    batch_size = 5128  


    for i in range(epoch):
        print(f"Epoch {i+1}/{epoch}")

        lstm.forward(X_train) 
        H = np.array(lstm.H)
        H = H.reshape((H.shape[0], H.shape[1])) 

        dense1.forward(H[1:, :])
        dense2.forward(dense1.output)

        y_hat = dense2.output
        
        y_train1 = y_train1[:y_hat.shape[0]].reshape(-1, 1)

        dy = y_hat - y_train1

        L = 0.5 * np.mean(dy**2)
        monitor[i] = L
        

        dense2.backward(dy)
        dense1.backward(dense2.dinput)
        lstm.backward(dense1.dinput)


        dense1.weights -= lr * dense1.dweights
        dense1.biases -= lr * dense1.dbiases
        dense2.weights -= lr * dense2.dweights
        dense2.biases -= lr * dense2.dbiases

        lstm.Uf -= lr * lstm.dUf
        lstm.Ui -= lr * lstm.dUi
        lstm.Uo -= lr * lstm.dUo
        lstm.Ug -= lr * lstm.dUg
        lstm.Wf -= lr * lstm.dWf
        lstm.Wi -= lr * lstm.dWi
        lstm.Wo -= lr * lstm.dWo
        lstm.Wg -= lr * lstm.dWg
        lstm.bf -= lr * lstm.dbf
        lstm.bi -= lr * lstm.dbi
        lstm.bo -= lr * lstm.dbo
        lstm.bg -= lr * lstm.dbg


        print(f'Current MSSE = {L}')
    

main()