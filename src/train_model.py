import pandas as pd
import numpy as np
import os
from joblib import dump
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None

from controllers import load_controller
from controllers.model_controller import PredictionModel
from mappings import DataTuple, MONTH, RENT_BASE, UTILITIES
from logger import logger

INPUT_DATA_FILEPATH = os.path.join(os.path.dirname(__file__), 'input_data/input_data.csv')
BASE_RENT_MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), 'model/base_rent_model.joblib')
UTILITIES_MODEL_FILEPATH = os.path.join(os.path.dirname(__file__), 'model/utilities_model.joblib')


def get_input_data():
    """
    replace this function with a database call to fetch new data for retraining
    """
    return load_controller.load_from_csv(INPUT_DATA_FILEPATH)


def update_model(cv_score=True):
    logger.debug('Initializing model update')

    logger.debug('Loading input data')
    data: DataTuple
    data = load_controller.frame_to_features_and_targets(get_input_data())
    logger.debug('Input data loaded')

    if cv_score:
        _ = cross_validate(data)

    logger.debug('Fitting regressors for base rent and utilities')
    model = PredictionModel()
    model.fit(data)
    logger.debug('Regressors has been fitted.')
    logger.debug('Base rent model R2 = {}'.format(model.base_rent_model.model_score))
    logger.debug('Utilities model R2 = {}'.format(model.utilities_model.model_score))

    logger.debug('Saving models')
    dump(model.base_rent_model, BASE_RENT_MODEL_FILEPATH)
    dump(model.utilities_model, UTILITIES_MODEL_FILEPATH)
    logger.debug('Models have been saved')


def cross_validate(data=None, show_plots=False):
    """
    calculates model scores and test its performance against baseline models
    first baseline model: simple seasonal model, apartment agnostic
    second baseline model: uses seasonal trend to predict the rate of change for the apartment price
    outputs rmse scores for 2 baseline models and the main model
    """
    logger.debug('Initializing cross validation')

    if not data:
        logger.debug('Loading input data')
        data: DataTuple
        data = load_controller.frame_to_features_and_targets(get_input_data())
        logger.debug('Input data loaded')

    targets = [RENT_BASE, UTILITIES]

    # do cross validation
    # 1. identify test samples: duplicated rows that can be traced in time for at least 6 months
    features = data.X.copy()
    features['id'] = features.groupby([c for c in features.columns if c != MONTH]).ngroup()
    ids_counts = features.groupby('id')['month'].nunique().sort_values(ascending=False)
    test_set_ids = ids_counts[ids_counts >= 5].index.tolist()

    # 2. separate test and train sets
    X_test = []
    base_rent_test = []
    utilities_test = []
    target_test = []
    features[RENT_BASE] = data.base_rent
    features[UTILITIES] = data.utilities

    for i in test_set_ids:
        X_test.append(data.X[features.id == i].iloc[0, :].to_frame().T)
        y_test = {}
        for t in targets:
            # mean monthly price
            monthly_prices = features[features.id == i][[t, MONTH]].groupby(MONTH).mean()
            # add missing months
            monthly_prices = monthly_prices.reindex(
                np.arange(monthly_prices.index.min(), monthly_prices.index.max() + 1),
                fill_value=np.nan)
            # fill missing price with previous month
            y_test[t] = monthly_prices.fillna(method='backfill')

        # starting prices for the first month
        base_rent_test.append(y_test[RENT_BASE].iloc[0, 0])
        utilities_test.append(y_test[UTILITIES].iloc[0, 0])
        # total rent for each month
        df = pd.merge(y_test[RENT_BASE], y_test[UTILITIES], left_index=True, right_index=True)
        target_test.append(df[RENT_BASE] + df[UTILITIES])

    X_train = data.X[~features.id.isin(test_set_ids)].reset_index(drop=True)
    y_train = {}
    y_train[RENT_BASE] = data.base_rent[~features.id.isin(test_set_ids)].reset_index(drop=True)
    y_train[UTILITIES] = data.utilities[~features.id.isin(test_set_ids)].reset_index(drop=True)

    # 3. calculate seasonal trend for baseline model
    seasonal_df = features[[MONTH]]
    seasonal_df['total_rent'] = features[RENT_BASE] + features[UTILITIES]
    seasonal_trend = seasonal_df.groupby(MONTH)['total_rent'].mean()

    # 4. fit the model to train set
    model = PredictionModel()
    model.fit(DataTuple(X=X_train, base_rent=y_train[RENT_BASE], utilities=y_train[UTILITIES]),
              cv_score=False)
    # logger.debug('Base rent model R2 = {}'.format(model.base_rent_model.model_score))
    # logger.debug('Utilities model R2 = {}'.format(model.utilities_model.model_score))

    # 5. predict for test set. y_test is only the price of the first month
    y_pred = []
    for sample in zip(X_test, base_rent_test, utilities_test):
        prediction = model.predict(DataTuple(*sample))
        y_pred.append(prediction)

    # 6. calculate seasonal baseline
    seasonal_pred = []
    for test_sample in X_test:
        next_month = test_sample[MONTH].iloc[0]
        # get price for the first month
        pred = []
        # get prices for the next three months
        for i in range(6):
            next_month = next_month + 1 if next_month < 12 else 1
            pred.append(seasonal_trend.loc[next_month])
        seasonal_pred.append(pred[:])

    # 7. calculate seasonal trend based baseline
    # Here we use seasonal trend to infer the rat of price change and then apply it to the current price
    seasonal_trend_pred = []
    for i, test_sample in enumerate(X_test):
        start_month = test_sample[MONTH].iloc[0]

        # take seasonal price of the starting month
        original_price = seasonal_trend.loc[start_month]

        # calculate rate of change based on the seasonal trend
        change_rates = [p / original_price for p in seasonal_pred[i]]

        # apply this rate to the starting price of the test sample
        pred = [target_test[i].loc[start_month] * r for r in change_rates]
        seasonal_trend_pred.append(pred)

    # 8. compare
    results = []
    rmse = {
        'predicted_price': [],
        'seasonal_baseline': [],
        'seasonal_trend_baseline': []
    }
    for i in range(len(X_test)):
        timeline = target_test[i][1:7].index
        results.append(pd.DataFrame(
            {
                'actual_price': target_test[i][1:7],
                'predicted_price': y_pred[i][:6],
                'seasonal_baseline': seasonal_pred[i][:6],
                'seasonal_trend_baseline': seasonal_trend_pred[i][:6]
            },
            index=timeline
        ))

        if show_plots:
            results[-1].plot(figsize=(20, 5))
            plt.show()

        for k in rmse.keys():
            rmse[k].append(
                ((results[-1][k] - results[-1]['actual_price']) ** 2).mean() ** 0.5)

    rmse = {k: np.mean(v) for k, v in rmse.items()}

    logger.debug('Cross validation finished')
    logger.debug(rmse)
    return rmse


if __name__ == '__main__':
    # update_model()
    cross_validate()
