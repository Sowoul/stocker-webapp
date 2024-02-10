# Stock Price Predictor

This project predicts stock prices using historical data and LSTM neural networks. It utilizes Python libraries such as NumPy, Matplotlib, Pandas Datareader, Flask, and Keras.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Stock Price Predictor is a Python application that uses historical stock price data to train an LSTM (Long Short-Term Memory) neural network model. This model can then predict future stock prices based on the provided historical data.

## Installation

To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Sowoul/stocker-webapp.git
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Start the Flask application:

    ```bash
    python app.py
    ```

    ALTERNATIVELY:
   
1. Clone the repository:

    ```bash
    git clone https://github.com/Sowoul/stocker-webapp.git
    ```
2. Run the app.bat batch file, it will install the necessary libraries, and then run the python file on its own, since its a batch file.



## Usage

1. Open the application in your web browser.
2. Open a web browser and go to http://localhost:8080.
3. Enter the name of the company whose stock price you want to predict in the search bar.
4. View the predicted stock price plotted alongside historical data.


## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request.
