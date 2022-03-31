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
    df = pd.read_csv('historical.csv')
    print(df.head())
    df.drop('Wolumen', axis='columns', inplace=True)
    df.drop('Zamkniecie', axis='columns', inplace=True)
    df.drop('Najnizszy', axis='columns', inplace=True)
    df.drop('Najwyzszy', axis='columns', inplace=True)
    print(df.head())
    return df


def calculate_ema(df: pd.DataFrame, n: int) -> float:
    alpha = 2 / (n + 1)
    return alpha


def main():
    account = Account(0, 1000)
    df = load_data()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

