import pytest

from tests.helpers import get_random_sample_dict, get_random_dataset_list, get_random_sample_vector
from controllers import load_controller
from controllers.model_controller import PredictionModel, BaseRegressorModel, ModelContainer


@pytest.fixture()
def dataset():
    sample_dataset = get_random_dataset_list()
    dataset = load_controller.load_from_list(sample_dataset)
    yield load_controller.frame_to_features_and_targets(dataset)


class TestModel:

    def test_base_rent_regressor(self, dataset):
        base_rent_model = BaseRegressorModel()
        base_rent_model.fit(dataset.X, dataset.base_rent)
        assert-1 <= base_rent_model.model_score <= 1

    def test_utilitiest_regressor(self, dataset):
        utilities_model = BaseRegressorModel()
        utilities_model.fit(dataset.X, dataset.utilities)
        assert -1 <= utilities_model.model_score <= 1

    def test_transformers(self, dataset):
        model = BaseRegressorModel()
        model.fit(dataset.X, dataset.base_rent)
        transformer = model.get_pipeline().steps[0][1]
        sample_vector = get_random_sample_vector()
        transformed_sample = transformer.transform(sample_vector)[0]
        assert len(transformed_sample) == 37

    def test_prediction_model(self, dataset):
        model = PredictionModel()
        model.initialize_model(base_rent_model=BaseRegressorModel().fit(dataset.X, dataset.utilities),
                               utilities_model=BaseRegressorModel().fit(dataset.X, dataset.utilities))
        sample = get_random_sample_dict()
        prediction = model.predict(load_controller.raw_sample_to_features(sample))
        assert len(prediction) == 6

    def test_model_container(self, dataset):
        container = ModelContainer(base_rent_model=BaseRegressorModel().fit(dataset.X, dataset.utilities),
                                   utilities_model=BaseRegressorModel().fit(dataset.X, dataset.utilities))

        sample = get_random_sample_dict()
        prediction = container.predict(sample)
        assert len(prediction) == 6
