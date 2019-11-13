import falcon
from wsgiref import simple_server

from controllers.predict_controller import PredictController, InfoController, ModelUpdateController
from controllers.model_controller import ModelContainer


def create_api(model_container=None):
    api = falcon.API()

    info_controller = InfoController()

    if not model_container:
        model_container = ModelContainer()

    predict_controller = PredictController(model_container)
    model_update_controller = ModelUpdateController(model_container)

    api.add_route('/predict', predict_controller)
    api.add_route('/update', model_update_controller)
    api.add_route('/', info_controller)

    return api


api = create_api()

if __name__ == '__main__':
    httpd = simple_server.make_server('127.0.0.1', 8000, api)
    httpd.serve_forever()