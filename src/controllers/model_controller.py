from joblib import load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd
import os
import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, RegressorMixin

from mappings import *
from logger import logger
from . import load_controller


class ModelContainer:
    def __init__(self, base_rent_model=None, utilities_model=None):
        self.base_rent_filepath = os.path.join(os.path.dirname(__file__), '..', 'model/base_rent_model.joblib')
        self.utilities_filepath = os.path.join(os.path.dirname(__file__), '..', 'model/utilities_model.joblib')

        if not base_rent_model:
            base_rent_model = self._load_base_rent_model()
        if not utilities_model:
            utilities_model = self._load_utilities_model()

        self.model = PredictionModel()
        self.model.initialize_model(base_rent_model, utilities_model)

        self._model_timestamp = datetime.datetime.now()
        self._needs_update = False

    def _load_base_rent_model(self):
        return load(self.base_rent_filepath)

    def _load_utilities_model(self):
        return load(self.utilities_filepath)

    def _update_model(self):
        logger.debug('Model update started')
        self.model.update_regressors(base_rent_regressor=self._load_base_rent_model(),
                                     utilities_regressor=self._load_utilities_model())

        self._needs_update = False
        self._model_timestamp = datetime.datetime.now()
        logger.debug('Model update finished')

    def mark_model_for_update(self):
        self._needs_update = True

    def predict(self, raw_data):
        """
        In order to update the model self.need_update is set to True
        Then during the next prediction the model will be updated
        """
        if self._needs_update:
            self._update_model()
            self._needs_update = False

        logger.debug('Extracting features')
        data = load_controller.raw_sample_to_features(raw_data)

        logger.debug('Executing prediction')
        prediction = self.model.predict(data)

        logger.debug('Prediction finished successfully: {}'.format(prediction))
        return prediction

    @property
    def model_timestamp(self):
        return str(self._model_timestamp)

    def get_model_info(self):
        return 'Model was last updated on {}'.format(self.model_timestamp)


class PredictionModel:
    def __init__(self):
        self.base_rent_model = BaseRegressorModel()
        self.utilities_model = BaseRegressorModel()

    def initialize_model(self, base_rent_model, utilities_model):
        self.base_rent_model = base_rent_model
        self.utilities_model = utilities_model

    def fit(self, data: DataTuple, cv_score=True):
        self.base_rent_model.fit(data.X, data.base_rent, cv_score)
        self.utilities_model.fit(data.X, data.utilities, cv_score)

    def predict(self, data: DataTuple, num_months=6):
        """
        The main idea is to do prediction in two general steps:
        1. predict total rent for the next 6 months. The absolute values are not very useful since they can shift
        from one year to another. However this also gives us a trend of how the rent will change
        2. Use this trend and apply it to an actual input rent value to get the prediction

        total rent is modeled as a sum of base rent and utilities bill (rent_total - rent_base)
        both these values can be affected by different factors throughout the year so should be modelled separately
        """
        # generate samples to get a time series: same flat but incrementing month
        data_points = pd.concat([data.X]*(num_months + 1), ignore_index=True)
        data_points[MONTH] = data_points[MONTH] + list(data_points.index)
        data_points[MONTH] = data_points[MONTH].apply(lambda x: x if x < 13 else x - 12)

        # predict base rent and utilities absolutes based on the base model (Random Forest Regressor)
        predicted_base_rent = self.base_rent_model.predict(data_points)
        predicted_utilities = self.utilities_model.predict(data_points)

        # extract rate of change from the prediction model
        change_rate_base_rent = [r / predicted_base_rent[0] for r in predicted_base_rent[1:]]
        change_rate_utilities = [u / predicted_utilities[0] for u in predicted_utilities[1:]]

        # apply rate of change to the initial flat price
        final_base_rent = [data.base_rent * rate for rate in change_rate_base_rent]
        final_utilities = [data.utilities * rate for rate in change_rate_utilities]

        # return total rent
        return [sum(x) for x in zip(final_base_rent, final_utilities)]

    def update_regressors(self, base_rent_regressor, utilities_regressor):
        self.base_rent_model = base_rent_regressor
        self.utilities_model = utilities_regressor


class BaseRegressorModel(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline=None):
        if pipeline:
            self._pipeline = pipeline
        else:
            self._pipeline = None
            self.init_pipeline()
        self._score = None

    def init_pipeline(self):
        preprocess = make_column_transformer(
            (OrdinalEncoder([FLAT_AGE_OPTIONS]), [FLAT_AGE]),
            (OneHotEncoder(handle_unknown='ignore', sparse=False), ONE_HOT_FEATURES),
            (OrdinalEncoder(), BOOLEAN_FEATURES),
            ('passthrough', USE_AS_IS_FEATURES)
        )
        forest_params = dict(
            n_estimators=100,
            random_state=42,
            max_depth=20,
            max_features='sqrt'
        )
        regressor = RandomForestRegressor(**forest_params)
        self._pipeline = make_pipeline(preprocess, regressor)

    def score_base_model(self, X, y):
        self._score = cross_val_score(self._pipeline, X, y, cv=5).mean()

    def get_pipeline(self):
        return self._pipeline

    def fit(self, X, y, cv_score=True):
        if cv_score:
            self.score_base_model(X, y)
        return self._pipeline.fit(X, y)

    def predict(self, X):
        return self._pipeline.predict(X)

    @property
    def model_score(self):
        return self._score









