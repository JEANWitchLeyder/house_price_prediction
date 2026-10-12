House Price Prediction

A small neural-network regression project that predicts house sale prices from the Ames, Iowa housing data, with a linear regression baseline for comparison.

Notebook: house_price_prediction.ipynb

Dataset: Ames Housing data (De Cock, 2011), covering sales from 2006 to 2010. Prices are in USD.

Approach
Select a small set of features, for example living area, rooms above grade, bathrooms, year built, overall quality, garage capacity and neighborhood.
Split the data, then fit the preprocessing on the training set only to avoid leakage.
Train a baseline linear regression model.
Train a Keras Sequential neural network with EarlyStopping, restoring the best weights.
Compare both models on held-out data (actual vs. predicted price).
Save the model and preprocessing, reload them, and check that predictions match.
