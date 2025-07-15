import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the Ridge regression model and the standard scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Collecting form data
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scaling input data
            new_data_scaled = standard_scaler.transform(
                [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
            )

            # Making prediction
            result = ridge_model.predict(new_data_scaled)

            # Rendering result on the home page
            return render_template('home.html', result=round(result[0], 2))
        
        except Exception as e:
            return f"Error occurred: {e}"

    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.environ.get("PORT", 5000))
