import random
from falcon import API

from controllers.load_controller import load_from_list
from controllers.model_controller import ModelContainer
from controllers.predict_controller import PredictController, InfoController
from mappings import *


def get_sample_dict():
    return {'date': '2018-09-15',
            'cnt_rooms': 3,
            'flat_area': 66.49,
            'rent_base': 742.69,
            'rent_total': 925.69,
            'flat_type': 'ground_floor',
            'flat_interior_quality': 'average',
            'flat_condition': 'good',
            'flat_age': '60+',
            'flat_thermal_characteristic': 121.0,
            'has_elevator': 'f',
            'has_balcony': 't',
            'has_garden': 'f',
            'has_kitchen': 't',
            'has_guesttoilet': 'f',
            'geo_city': 'hamburg',
            'geo_city_part': 'mitte'}


def get_random_sample_dict():
    base_rent = random.random()*5000
    return {'date': '2018-{}-15'.format(random.randint(1, 12)),
            'cnt_rooms': random.randint(1, 5),
            'flat_area': random.random()*100,
            'rent_base': base_rent,
            'rent_total': base_rent + random.random()*500,
            'flat_type': random.choice(FLAT_TYPE_OPTIONS),
            'flat_interior_quality': random.choice(FLAT_INTERIOR_QUALITY_OPTIONS),
            'flat_condition': random.choice(FLAT_CONDITION_OPTIONS),
            'flat_age': random.choice(FLAT_AGE_OPTIONS),
            'flat_thermal_characteristic': random.random()*300,
            'has_elevator': random.choice(['t', 'f']),
            'has_balcony': random.choice(['t', 'f']),
            'has_garden': random.choice(['t', 'f']),
            'has_kitchen': random.choice(['t', 'f']),
            'has_guesttoilet': random.choice(['t', 'f']),
            'geo_city': 'hamburg',
            'geo_city_part': random.choice(GEO_CITY_PART_OPTIONS)
            }


def get_random_dataset_list(length=100):
    return [get_random_sample_dict() for i in range(length)]


def get_random_sample_vector():
    return load_from_list([get_random_sample_dict()])


def create_api(model_container=None):
    api = API()

    if not model_container:
        # model_container instance
        model_container = ModelContainer()

    # handling requests
    api.add_route('/predict', PredictController(model_container))
    api.add_route('/info', InfoController())
    return api


