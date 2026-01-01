from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# ------------------------------
# Load Model, Scaler & Encoder Safely
# ------------------------------
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'minmaxscaler.pkl'
ENCODER_PATH = 'label_encoder.pkl'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found!")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("minmaxscaler.pkl not found!")

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("label_encoder.pkl not found!")

model = pickle.load(open(MODEL_PATH, 'rb'))
ms = pickle.load(open(SCALER_PATH, 'rb'))
le = pickle.load(open(ENCODER_PATH, 'rb'))   # ✅ VERY IMPORTANT FIX

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH'])
        rainfall = float(request.form['Rainfall'])

        # Check for negative values
        if any(v < 0 for v in [N, P, K, temp, humidity, ph, rainfall]):
            return render_template('index.html',
                                   result="⚠ All values must be non-negative!")

        # Prepare and scale features
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        scaled = ms.transform(features)

        # ✅ Predict + Decode Crop Name CORRECTLY
        prediction = model.predict(scaled)[0]
        crop_name = le.inverse_transform([prediction])[0]

        # Show clean output
        result = f"🌾 {crop_name} is the best crop to cultivate here!"

        return render_template('index.html', result=result)

    except ValueError:
        return render_template('index.html',
                               result="⚠ Please enter valid numbers in all fields.")
    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
