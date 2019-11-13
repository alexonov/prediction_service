from collections import namedtuple

DataTuple = namedtuple('DataTuple', 'X base_rent utilities')

# categorical options
FLAT_TYPE_OPTIONS = ['apartment', 'roof_storey', 'ground_floor',
                     'terraced_flat', 'maisonette', 'raised_ground_floor',
                     'half_basement', 'penthouse', 'loft']
FLAT_INTERIOR_QUALITY_OPTIONS = ['average', 'sophisticated', 'normal', 'simple', 'luxury']
FLAT_CONDITION_OPTIONS = ['good', 'first_time_use', 'mint_condition',
                          'first_time_use_after_refurbishment', 'renovated', 'mediocre']
FLAT_AGE_OPTIONS = ['<1', '<5', '<10', '<20', '<30', '<40', '<50', '<60', '60+']
GEO_CITY_PART_OPTIONS = ['wandsbek', 'mitte', 'altona', 'hamburg-nord', 'eimsbuettel',
                         'bergedorf', 'harburg']

# feature names
HAS_ELEVATOR = 'has_elevator'
HAS_BALCONY = 'has_balcony'
HAS_GARDEN = 'has_garden'
HAS_KITCHEN = 'has_kitchen'
HAS_GUESTTOILET = 'has_guesttoilet'
FLAT_TYPE = 'flat_type'
FLAT_INTERIOR_QUALITY = 'flat_interior_quality'
FLAT_CONDITION = 'flat_condition'
GEO_CITY_PART = 'geo_city_part'
FLAT_AREA = 'flat_area'
FLAT_THERMAL_CHARACTERISTIC = 'flat_thermal_characteristic'
RENT_TOTAL = 'rent_total'
RENT_BASE = 'rent_base'
FLAT_AGE = 'flat_age'
CNT_ROOMS = 'cnt_rooms'
MONTH = 'month'

UTILITIES = 'utilities'

USED_FEATURES = [
    HAS_ELEVATOR,
    HAS_BALCONY,
    HAS_GARDEN,
    HAS_KITCHEN,
    HAS_GUESTTOILET,
    FLAT_TYPE,
    FLAT_INTERIOR_QUALITY,
    FLAT_CONDITION,
    GEO_CITY_PART,
    FLAT_AREA,
    FLAT_THERMAL_CHARACTERISTIC,
    FLAT_AGE,
    CNT_ROOMS,
    MONTH
]

BOOLEAN_FEATURES = [
    HAS_ELEVATOR,
    HAS_BALCONY,
    HAS_GARDEN,
    HAS_KITCHEN,
    HAS_GUESTTOILET,
]

ONE_HOT_FEATURES = [
    FLAT_TYPE,
    FLAT_INTERIOR_QUALITY,
    FLAT_CONDITION,
    GEO_CITY_PART,
]

USE_AS_IS_FEATURES = [
    FLAT_AREA,
    CNT_ROOMS,
    FLAT_THERMAL_CHARACTERISTIC,
    MONTH
]
