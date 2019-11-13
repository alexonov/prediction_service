import falcon
from marshmallow import ValidationError

from schemas import PredictRequestSchema, PredictResponseSchema
from logger import logger
from train_model import update_model
from controllers.model_controller import ModelContainer


class InfoController:
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = ('\nThis is an API for a rent price prediction model for flats in Hamburg.\n'
                     'The model takes apartment parameters and current rent amount and predicts the rent for next 6 months.\n'
                     'To learn more about this model, send a GET request to the /predict endpoint \n')


class ModelUpdateController:
    def __init__(self, model_container: ModelContainer):
        self.model = model_container

    def on_get(self, req, resp):
        action = req.params.get('action', None)
        if action == 'update':
            try:
                # update model
                logger.info('Updating model')
                # TODO: run training process asynchronously
                update_model()
                logger.info('Model update finished')

                # mark currently loaded model for update so that it will be reloaded before the next prediction
                self.model.mark_model_for_update()
                resp.body = 'Model has been retrained'
                resp.status = falcon.HTTP_200
            except Exception as e:
                logger.error(e)
                resp.status = falcon.HTTP_500
                resp.media = e

        elif action == 'status':
            resp.body = 'Last update at: {}'.format(self.model.model_timestamp)
            resp.status = falcon.HTTP_200
        else:
            resp.body = ('This endpoint is used for updating the model.\n'
                         'Send a request action=update to retrain the model\n'
                         'or action=status to check the last time the model was updated\n\n')
            resp.status = falcon.HTTP_200


class PredictController:
    def __init__(self, model_container: ModelContainer):
        self.model = model_container

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = ('\nThis is the PREDICT endpoint. \n'
                     'Both requests and responses are served in JSON. \n'
                     '\n'
                     'INPUT: \n'
                     'date (date): date when the ad was published\n'
                     'cnt_rooms (int): number of rooms in the flat\n'
                     'flat_area (float): living area of the flat (in square meters)\n'
                     'rent_base (float): base monthly rent for the flat (in euro)\n'
                     'rent_total (float): total (including utilities) monthly rent for the flat (in euro)\n'
                     'flat_type (string): type of the property, e.g. apartment, roof_storey, etc.\n'
                     'flat_interior_quality (string): quality of the flat interior\n'
                     'flat_condition (string): flat condition, e.g. normal, good, etc.\n'
                     'flat_age (string): category of the flat\'s age (in years), e.g. <5, <10, ..., <50 etc.\n'
                     'flat_thermal_characteristic (float): energy consumption for the flat (in kWh per square meter per year)\n'
                     'has_elevator (boolean): indicates if the house has an elevator\n'
                     'has_balcony (boolean): indicates if the flat has a balcony\n'
                     'has_garden (boolean): indicates if a garden can be accessed from the flat\n'
                     'has_kitchen (boolean): indicates if the flat has built-in kitchen\n'
                     'has_guesttoilet (boolean): indicates if the flat has guest toilet\n'
                     'geo_city (string): city location of the flat\n'
                     'geo_city_part (string): city district location of the flat\n'
                     'OUTPUT: Prediction (total rent)   \n'
                     '   "Total rent": [...]       \n\n'
                     'To update the model send get request with action=update\n'
                     'To see status of the current model send get request with action=status\n')

    def on_post(self, request: falcon.Request, response: falcon.Response):
        try:
            validated = PredictRequestSchema().load(request.media)
        except ValidationError:
            logger.error('Invalid request schema')
            raise falcon.HTTPBadRequest('Invalid request schema')
        try:
            predicted_data = self.model.predict(validated)
        except Exception as e:
            logger.error(e)
            raise falcon.HTTP_500

        output = {'prediction': predicted_data}

        try:
            response_data = PredictResponseSchema().dump(output)
            response.status = falcon.HTTP_200
            response.media = response_data
        except ValidationError:
            logger.error('Invalid request schema')
            response.status = falcon.HTTP_500
            response.media = ValidationError
