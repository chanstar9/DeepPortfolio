from tqdm import tqdm
import datetime as dt
import pandas as pd

from Phases import *


def backtester(start_date: dt.datetime, train_period: int, test_period: int, auto_encoding_param, param,
               index_name='코스피'):
    """
    :param start_date: backtest start date
    :param train_period: period of market information
    :param test_period: rebalance period
    :param auto_encoding_param:
    :param param:
    :param index_name: target to replicate
    :return:
    """
    # initial setting
    company_total = pd.read_csv('data/company_total.csv')
    index_total = pd.read_csv('data/index_total.csv')
    company_total[DATE] = pd.to_datetime(company_total[DATE], format='%Y-%m-%d')
    index_total[DATE] = pd.to_datetime(index_total[DATE], format='%Y-%m-%d')
    trade_dates = pd.DataFrame(np.unique(company_total[DATE].values), columns=[DATE])

    # if start_date not in trade_dates:
    #     raise ValueError('start date is not business day or out of bound')
    # if index_name not in index_total[NAME]:
    #     raise ValueError('index is not in data')

    index_total = index_total[index_total[NAME] == index_name]
    if index_name != '코스피':
        company_total = company_total[company_total[SECTOR] == index_name]
    backtesting_dates = trade_dates[trade_dates >= start_date].dropna().values.reshape(1, -1)[0][::test_period]
    for date in tqdm(backtesting_dates):
        # setting data
        train_start_date = trade_dates.iloc[np.where(trade_dates == date)[0][0] - train_period][0]
        test_end_date = trade_dates.iloc[np.where(trade_dates == date)[0][0] + test_period][0]
        filtered_company = company_total[
            (company_total[DATE] >= train_start_date) & (company_total[DATE] < test_end_date)]
        filtered_index = index_total[(index_total[DATE] >= train_start_date) & (index_total[DATE] < test_end_date)]

        # stock
        company_cp = pd.pivot_table(filtered_company, index=DATE, columns=NAME, values=CLOSE_P).dropna(axis=1,
                                                                                                       how='any')
        company_ret = pd.pivot_table(filtered_company, index=DATE, columns=NAME, values=RET).dropna(axis=1, how='any')
        company_vol = pd.pivot_table(filtered_company, index=DATE, columns=NAME, values=VOLUME).dropna(axis=1,
                                                                                                       how='any')

        stock = defaultdict(defaultdict)
        stock[CALIBRATE][CLOSE_P] = company_cp[(company_cp.index >= train_start_date) & (company_cp.index < date)]
        stock[CALIBRATE][RET] = company_ret[(company_ret.index >= train_start_date) & (company_ret.index < date)]
        stock[CALIBRATE][VOLUME] = company_vol[(company_vol.index >= train_start_date) & (company_vol.index < date)]
        stock[VALIDATE][CLOSE_P] = company_cp[(company_cp.index >= date) & (company_cp.index < test_end_date)]
        stock[VALIDATE][RET] = company_ret[(company_ret.index >= date) & (company_ret.index < test_end_date)]
        stock[VALIDATE][VOLUME] = company_vol[(company_vol.index >= date) & (company_vol.index < test_end_date)]

        # index
        index_cp = pd.pivot_table(filtered_index, index=DATE, columns=NAME, values=CLOSE_P).dropna(axis=1, how='any')
        index_ret = pd.pivot_table(filtered_index, index=DATE, columns=NAME, values=RET).dropna(axis=1, how='any')

        index = defaultdict(defaultdict)
        index[CALIBRATE][CLOSE_P] = index_cp[(index_cp.index >= train_start_date) & (index_cp.index < date)]
        index[CALIBRATE][RET] = index_ret[(index_ret.index >= train_start_date) & (index_ret.index < date)]
        index[VALIDATE][CLOSE_P] = index_cp[(index_cp.index >= date) & (index_cp.index < test_end_date)]
        index[VALIDATE][RET] = index_ret[(index_ret.index >= date) & (index_ret.index < test_end_date)]

        # Phase 1: Autoencoding
        input_dim = len(stock[CALIBRATE][CLOSE_P].columns)
        encoding_dim, ranking = Phase1(input_dim,
                                       encode_activation=auto_encoding_param[ENCODE_ACTIVATION],
                                       decode_activation=auto_encoding_param[DECODE_ACTIVATION],
                                       optimizer=auto_encoding_param[OPTIMIZER],
                                       loss=auto_encoding_param[LOSS],
                                       shuffle=auto_encoding_param[SHUFFLE],
                                       epoch=auto_encoding_param[EPOCH],
                                       batch_size=auto_encoding_param[BATCH_SIZE],
                                       validation_split=auto_encoding_param[VALIDATION_SPLIT],
                                       x_train=stock[CALIBRATE][RET]
                                       )

        # Phase 2, 3, 4: Calibrating, Validating, Verifying
        selected_stocks = Phase4(input_dim, encoding_dim, ranking, stock, index, stock[CALIBRATE][VOLUME],
                                 activation_i=param[ACTIVATION_I],
                                 activation_o=param[ACTIVATION_O],
                                 optimizer=param[OPTIMIZER],
                                 loss=param[LOSS],
                                 shuffle=param[SHUFFLE],
                                 epoch=param[EPOCH],
                                 batch_size=param[BATCH_SIZE],
                                 validation_split=param[VALIDATION_SPLIT])

        #
    return


if __name__ == '__main__':
    backtester(
        start_date=dt.datetime(2009, 1, 2),
        train_period=240 * 2,
        test_period=5,
        auto_encoding_param={
            ENCODE_ACTIVATION: 'sigmoid',
            DECODE_ACTIVATION: 'linear',
            OPTIMIZER: 'adam',
            LOSS: 'mse',
            SHUFFLE: False,
            EPOCH: 300,
            BATCH_SIZE: 50,
            VALIDATION_SPLIT: 0.1,
        },
        param={
            ACTIVATION_I: 'sigmoid',
            ACTIVATION_O: 'sigmoid',
            OPTIMIZER: 'adam',
            LOSS: 'mse',
            SHUFFLE: False,
            EPOCH: 300,
            BATCH_SIZE: 50,
            VALIDATION_SPLIT: 0.1,
        },
        index_name='코스피'
    )
