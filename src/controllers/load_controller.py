import pandas as pd

from mappings import *


def load_from_csv(filepath):
    data = pd.read_csv(filepath)
    return _process(data)


def load_from_list(data):
    data = pd.DataFrame(data)
    return _process(data)


def _process(dataset):
    dataset[FLAT_TYPE] = dataset[FLAT_TYPE].replace('appartment', 'apartment')
    dataset = dataset[[c for c in dataset.columns if c != 'geo_city']]
    dataset = dataset[dataset[RENT_TOTAL] != 0]
    dataset = dataset[dataset[RENT_TOTAL] >= dataset[RENT_BASE]]
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset[UTILITIES] = dataset[RENT_TOTAL] - dataset[RENT_BASE]
    dataset = dataset[dataset[UTILITIES] < 5000]
    dataset = dataset[dataset[UTILITIES] > 0]
    dataset[FLAT_THERMAL_CHARACTERISTIC].fillna(dataset[FLAT_THERMAL_CHARACTERISTIC].mean(), inplace=True)
    dataset[MONTH] = dataset.apply(lambda row: row['date'].month, axis=1)
    dataset = dataset.reset_index(drop=True)
    # reorder columns
    dataset = dataset[USED_FEATURES + [UTILITIES, RENT_BASE]]
    return dataset


def frame_to_features_and_targets(data):
    return DataTuple(X=data[USED_FEATURES], base_rent=data[RENT_BASE], utilities=data[UTILITIES])


def sample_to_features_and_targets(data):
    return DataTuple(X=data[USED_FEATURES], base_rent=data[RENT_BASE].values[0], utilities=data[UTILITIES].values[0])


def raw_sample_to_features(data):
    frame = load_from_list([data])
    return sample_to_features_and_targets(frame)
