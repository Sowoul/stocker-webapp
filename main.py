import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import yfinance as yfin
import base64
from io import BytesIO

from flask import Flask, render_template, request

yfin.pdr_override()
plt.style.use("fivethirtyeight")

# Load Data
def predict(company: str):
    start = dt.datetime(2012, 1, 1)
    end = dt.datetime.today()
    data = pdr.get_data_yahoo(company, start, end)
    dataframe = data[['Adj Close']]
    dataset = dataframe.values
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
    except:
        print('sad')
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    # Reshape into X=t and Y=t+1
    look_back = 20
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
    epoch_num = 50

    # Create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(8))
    model.add(Dense(16))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=epoch_num, batch_size=32, verbose=2)

    # Make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Invert predictions to original scale
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # Shift train predictions for plotting
    trainPredictPlot = np.empty_like(dataset)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(dataset)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

    # Make predictions for future time steps
    forecast_input = testX[-1]
    forecast_input = np.reshape(forecast_input, (1, look_back, 1))  # Reshape input

    # Generate predictions for the next 60 time steps
    num_predictions = 10
    forecast = []
    for i in range(num_predictions):
        next_pred = model.predict(forecast_input)
        forecast.append(next_pred[0][0])
        next_pred_reshaped = np.reshape(next_pred, (1, 1, 1))
        forecast_input = np.append(forecast_input[:, 1:, :], next_pred_reshaped, axis=1)

    # Reshape the forecast array to the correct shape
    forecast = np.reshape(forecast, (num_predictions, 1))

    # Inverse the forecasted values to the original scale
    forecast = scaler.inverse_transform(forecast)

    # Plot baseline and predictions
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(scaler.inverse_transform(dataset), label='True data')
    axs[0].plot(trainPredictPlot, label='Training predictions')
    testPredictPlot = np.concatenate((np.zeros((len(trainPredictPlot) - len(testPredictPlot), 1)), testPredictPlot))
    axs[0].plot(testPredictPlot, label='Test predictions')
    axs[0].plot(range(len(dataset), len(dataset) + num_predictions), forecast, label='Future predictions')
    axs[0].set_title(company + "'s Stock Price")
    axs[0].legend()

    axs[1].plot(scaler.inverse_transform(dataset), label='True data')
    axs[1].plot(trainPredictPlot, label='Training predictions')
    testPredictPlot = np.concatenate((np.zeros((len(trainPredictPlot) - len(testPredictPlot), 1)), testPredictPlot))
    axs[1].plot(testPredictPlot, label='Test predictions')
    axs[1].plot(range(len(dataset), len(dataset) + num_predictions), forecast, label='Future predictions')
    axs[1].set_xlim(len(scaler.inverse_transform(dataset)) - 100 + num_predictions,
                    len(scaler.inverse_transform(dataset)) + num_predictions + 5)

    # Save the plot as a data URL
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plot_url = f"data:image/png;base64,{plot_data}"

    return plot_url


app = Flask(__name__)


@app.route('/')
def search():
    company = request.args.get('search')
    plot_url = predict(company)
    return render_template('index.html', plot_url=plot_url)


if __name__ == '__main__':
    app.run(port=8080)
