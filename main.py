import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import typing


class Account:
    def __init__(self, cash, stocks):
        self.cash = cash
        self.stocks = stocks

    def calculate_profit(self, prices):
        return self.stocks * prices[1000] + self.cash/(1000 * prices[0])


def load_data():
    """
    loads data from csv file, only opening prices are taken
    returns a dataframe containing two columns: 'Date' and 'Zamkniecie' (closing price)
    """
    df = pd.read_csv('historical.csv')
    df.drop('Wolumen', axis='columns', inplace=True)
    df.drop('Otwarcie', axis='columns', inplace=True)
    df.drop('Najnizszy', axis='columns', inplace=True)
    df.drop('Najwyzszy', axis='columns', inplace=True)
    return df


def create_table(df: pd.core.frame.DataFrame):
    """
    creates appropriate slices for calculate_ema function,
    passes them to calculate_ema, and saved in df Dataframe.
    :param df: a pandas dataframe of size(X, ) containing historical prices of stock for calculations
    :return: None, the changes are made inplace for better memory management
    """

    short_term_periods = 11
    long_term_periods = 25
    signal_periods = 8
    prices = df['Zamkniecie']
    ema_12 = np.zeros(prices.shape[0]-long_term_periods, dtype='float64')
    ema_26 = np.zeros(prices.shape[0]-long_term_periods, dtype='float64')
    macd = np.zeros(prices.shape[0]-long_term_periods, dtype='float64')
    signal = np.zeros(prices.shape[0]-long_term_periods-signal_periods, dtype='float64')

    for i, k in enumerate(ema_12):
        ema_12[i] = calculate_ema(prices[i:i+short_term_periods])
        ema_26[i] = calculate_ema(prices[i:i+long_term_periods])
        macd[i] = ema_26[i] - ema_12[i]

    for i, k in enumerate(signal):
        signal[i] = calculate_ema(macd[i:i+signal_periods])

    df.drop(df.index[0:25], inplace=True)
    df['ema_12'] = ema_12
    df['ema_26'] = ema_26
    df['macd'] = ema_12 - ema_26
    df.drop(df.index[0:8], inplace=True)
    df['signal'] = signal
    print(df)


def calculate_ema(period: pd.core.frame.DataFrame) -> float:
    """    
    :param period: a slice of pandas Dataframe containing data from which to calculate the EMA
    :return: calculated value of EMA
    """
    alpha = 2 / (len(period) + 1)
    nominator = 0
    denominator = 0
    for i, k in enumerate(period):
        nominator = nominator + (((1 - alpha)**i) * k)
        denominator = denominator + ((1-alpha)**i)
    assert denominator != 0
    return nominator/denominator

def draw_plot(df: pd.core.frame.DataFrame):
    plt.plot(df['Data'], df['Zamkniecie'])
    plt.plot(df['Data'], df['macd'])
    plt.plot(df['Data'], df['signal'])
    plt.show()



def main():
    account = Account(0, 1000)
    df = load_data()
    print(df.head())
    create_table(df)
    draw_plot(df)




if __name__ == '__main__':
    main()

