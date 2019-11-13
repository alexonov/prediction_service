from marshmallow import Schema, validates_schema, ValidationError
from marshmallow.fields import Integer, String, Date, Float, List
from marshmallow.validate import Range, OneOf

from mappings import FLAT_TYPE_OPTIONS, FLAT_INTERIOR_QUALITY_OPTIONS, FLAT_CONDITION_OPTIONS, \
    FLAT_AGE_OPTIONS, GEO_CITY_PART_OPTIONS


class PredictRequestSchema(Schema):
    date = Date(required=True) # '2018-09-15',
    cnt_rooms = Integer(validate=[Range(min=1)], required=True) # 1,
    flat_area = Float(validate=[Range(min=0)], required=True) # 36.0,
    rent_base = Float(validate=[Range(min=0)], required=True) # 530.0,
    rent_total = Float(validate=[Range(min=0)], required=True) # 650.0,
    flat_type = String(validate=OneOf(FLAT_TYPE_OPTIONS), required=True) # 'apartment',
    flat_interior_quality = String(validate=OneOf(FLAT_INTERIOR_QUALITY_OPTIONS), required=True) # 'average',
    flat_condition = String(validate=OneOf(FLAT_CONDITION_OPTIONS), required=True) # 'good',
    flat_age = String(validate=OneOf(FLAT_AGE_OPTIONS), required=True) # '60+',
    flat_thermal_characteristic = Float(validate=[Range(min=0)], required=True) # nan,
    has_elevator = String(validate=OneOf(['t', 'f']), required=True) # 'f',
    has_balcony = String(validate=OneOf(['t', 'f']), required=True) # 't',
    has_garden = String(validate=OneOf(['t', 'f']), required=True) # 't',
    has_kitchen = String(validate=OneOf(['t', 'f']), required=True) # 't',
    has_guesttoilet = String(validate=OneOf(['t', 'f']), required=True) # 'f',
    geo_city = String(validate=OneOf(['hamburg']), required=True) # 'hamburg',
    geo_city_part = String(validate=OneOf(GEO_CITY_PART_OPTIONS), required=True) # 'wandsbek'

    @validates_schema
    def validate_numbers(self, data, **kwargs):
        if data["rent_base"] > data["rent_total"]:
            raise ValidationError("rent_base must be greater than rent_total")


class PredictResponseSchema(Schema):
    prediction = List(Float(validate=Range(min=0)), required=True)

