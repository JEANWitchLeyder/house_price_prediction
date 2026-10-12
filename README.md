House Price Prediction

A small neural-network regression project that predicts house sale prices from the Ames, Iowa housing data, with a linear regression baseline for comparison.

Notebook: house_price_prediction.ipynb

Dataset

Ames Housing data (De Cock, 2011), covering sales from 2006 to 2010. Prices are in USD.

Approach
Select a small set of features, for example living area, rooms above grade, bathrooms, year built, overall quality, garage capacity and neighborhood.
Split the data, then fit the preprocessing on the training set only to avoid leakage.
Train a baseline linear regression model.
Train a Keras Sequential neural network with EarlyStopping, restoring the best weights.
Compare both models on held-out data (actual vs. predicted price).
Save the model and preprocessing, reload them, and check that predictions match.
Saved Outputs
File	Contents
house_model.keras	Trained neural network
preprocessor.joblib	Fitted preprocessing pipeline
linear_baseline.joblib	Fitted linear baseline
schema.json	Feature list, price scale, currency and period
Usage
python
import json
import joblib
import pandas as pd
from keras.models import load_model

OUT = "outputs"  # folder containing the saved files

model = load_model(f"{OUT}/house_model.keras")
preprocessor = joblib.load(f"{OUT}/preprocessor.joblib")
schema = json.loads(open(f"{OUT}/schema.json").read())

house = pd.DataFrame([{...}])[schema["features"]]  # fill in the feature values
x = preprocessor.transform(house).astype("float32")
price = float(model(x, training=False).numpy().ravel()[0]) * schema["price_scale"]
print(f"Estimated price: USD {price:,.2f}")
Setup
bash
pip install pandas numpy scikit-learn tensorflow joblib
Notes
The model is trained on 2006–2010 Ames sales only, so it should not be used to price homes in other places or years.
The example prediction in the notebook is illustrative, not a validated valuation.
This is a learning project. The references do not show that neural networks outperform linear models on this data.
References
De Cock (2011), Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project, Journal of Statistics Education 19(3). https://doi.org/10.1080/10691898.2011.11889627
Keras and scikit-learn documentation for EarlyStopping, model saving and data-leakage guidance.
