from copy import deepcopy
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import innvestigate

from column import *
from Auto_encoding import *
from encoding_dim_func import *


def Phase1(input_dim, encode_activation, decode_activation, optimizer, loss, x_train, shuffle, epoch, batch_size,
           validation_split):
    # setting encoding dim
    encoding_dim = Find_factor_number(x_train, 0)

    # structuring auto-encoder
    autoencoder = Autoencoder(encoding_dim, input_dim, encode_activation, decode_activation, optimizer, loss,
                              x_train, shuffle, epoch, batch_size, validation_split)

    # get reconstructive
    reconstruct = autoencoder.predict(x_train)
    communal_information = []

    for i in range(0, len(x_train.columns)):
        diff = np.linalg.norm((x_train.iloc[:, i] - reconstruct[:, i]))  # 2 norm difference
        communal_information.append(float(diff))

    ranking = np.array(communal_information).argsort()
    return encoding_dim, ranking


def Phase2(non_communal, encoding_dim, ranking, stock, index, activation_i, activation_o, optimizer, loss, shuffle,
           epoch, batch_size, validation_split):
    dl_scaler = defaultdict(StandardScaler)

    s = 2 * non_communal
    stock_index = np.concatenate((ranking[:non_communal], ranking[-non_communal:]))  # portfolio index

    # connect all layers
    input_stock = Input(shape=(s,))
    input_layer = Dense(encoding_dim, activation=activation_i, kernel_regularizer=regularizers.l2(0.01))(input_stock)
    output_layer = Dense(1, activation=activation_o, kernel_regularizer=regularizers.l2(0.01))(input_layer)

    # construct and compile deep learning routine
    deep_learner = Model(input_stock, output_layer)
    deep_learner.compile(optimizer=optimizer, loss=loss)

    x_train = stock[CALIBRATE][RET].iloc[:, stock_index]
    y_train = index[CALIBRATE][RET]

    # Multi-layer Perceptron is sensitive to feature scaling, so it is highly recommended to scale your data
    dl_scaler[s] = StandardScaler()
    dl_scaler[s].fit(x_train)
    x_train = dl_scaler[s].fit_transform(x_train)

    deep_learner.fit(x_train, y_train,
                     shuffle=shuffle,
                     epochs=epoch,
                     batch_size=batch_size,
                     verbose=0,
                     callbacks=[EarlyStopping(patience=10)],
                     validation_split=validation_split
                     )  # fit the model
    return deep_learner


def Phase3(deep_learner, non_communal, ranking, stock, index, comparing_plot=False):
    dl_scaler = defaultdict(StandardScaler)

    s = 2 * non_communal
    stock_index = np.concatenate((ranking[:non_communal], ranking[-non_communal:]))  # portfolio index

    x_train = stock[CALIBRATE][RET].iloc[:, stock_index]
    x_train = dl_scaler[s].fit_transform(x_train)

    # get weight
    analyzer = innvestigate.create_analyzer("lrp.w_square", deep_learner)
    analysis = analyzer.analyze(x_train)

    # is it good?
    relative_percentage = deepcopy(deep_learner.predict(x_test))
    relative_percentage[0] = 0
    relative_percentage = (relative_percentage / 100) + 1

    index_predict = index[VALIDATE][CLOSE_P][0] * (relative_percentage.cumprod())
    total_2_norm_diff = np.linalg.norm((index_predict[VALIDATE][s] - index[VALIDATE][CLOSE_P]))

    # if comparing_plot:
    #     # comparing graph
    #     pd.Series(index['validate']['lp'].as_matrix(), index=pd.date_range(start='01/03/2014', periods=122, freq='W')).plot(
    #         label='IBB original', legend=True)
    #     for s in [25, 45, 65]:
    #         pd.Series(index_predict['validate'][s], index=pd.date_range(start='01/03/2014', periods=122, freq='W')).plot(
    #             label='IBB S' + str(s), legend=True)
    #         print("S" + str(s) + " 2-norm difference: ", total_2_norm_diff['validate'][s])
    return index_predict, total_2_norm_diff, dl_scaler


def Phase4(input_dim, encoding_dim, ranking, stock, index, stock_volume, activation_i, activation_o, optimizer, loss,
           shuffle, epoch, batch_size,
           validation_split, deep_frontier_plot=False):
    error = []
    for non_communal in range(5, input_dim, 5):
        deep_learner = Phase2(non_communal, encoding_dim, ranking, stock, index, activation_i, activation_o, optimizer,
                              loss, shuffle, epoch, batch_size, validation_split)
        _, total_2_norm_diff, _ = Phase3(deep_learner, non_communal, ranking, stock, index)
        error.append(total_2_norm_diff * non_communal)
    criteria = np.array(error) / stock_volume

    if deep_frontier_plot:
        # Plot Efficient Deep Frontier
        mse = [e / len(index['validate']['lp']) for e in error]  # mse = sum of 2 norm difference/ # of test dates
        plt.gca().invert_yaxis()
        plt.plot(mse, list(range(5, 79, 1)))
        plt.xlabel('Mean Squared Error')
        plt.ylabel('number of stocks in the portfolio')
        plt.show()
    return np.argmin(criteria) + 5
