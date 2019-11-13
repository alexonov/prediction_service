# Prediction API

Total rent prediction API

### Run prediction service
```
docker build -t prediction-app .
docker run -p 9000:9000 prediction-app
```

### Run unittests
```
cd src
export PYTHONPATH=$(pwd)
pytest -s tests
```

### Available endpoints

`/` [GET] - returns general info about the API

`/predict` [GET] - returns required format

`/predict` [POST] - predicts total rent for a given apartment for the next 6 monts

`/update?action=status` [GET] - returns the timestamp of the last model update

`/update?action=update` [GET] - initiated model update

### Report
Please find report in the `report.ipynb` file