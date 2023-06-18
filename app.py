from flask import Flask, request
import joblib
import numpy

MODEL_PATH = 'ml_model/model.pkl'
SCALER_X_PATH = 'ml_model/scaler_x.pkl'
SCALER_Y_PATH = 'ml_model/scaler_y.pkl'

app = Flask(__name__)
model = joblib.load(MODEL_PATH)
sc_x = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

@app.route('/predict_price', methods = ['GET'])
def predict():
    arg = request.args
    open_plan = arg.get('open_plan', default=-1, type=int)
    rooms = arg.get('rooms', default=-1, type=int)
    area = arg.get('area', default=-1, type=float)
    renovation = arg.get('renovation', default=-1, type=int)

    x = numpy.array([open_plan,rooms, area, renovation]).reshape(1, -1)
    x = sc_x.transform(x)
    result = model.predict(x)
    result = sc_y.inverse_tranform(result.reshape(1, -1))

    return str(result[0][0])

def newfunc():
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5444, host='0.0.0.0')