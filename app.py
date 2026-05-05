from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

app = Flask(__name__)

# Load dataset
df = pd.read_csv('car dataset.csv')

# Drop duplicates
df.drop_duplicates(inplace=True)

# Encode categorical variables
df_processed = pd.get_dummies(df, drop_first=True)

# Features and target
X = df_processed.drop('selling_price', axis=1)
y = df_processed['selling_price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
numerical_cols = ['year', 'km_driven']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Train XGBoost
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        year = int(request.form['year'])
        km_driven = float(request.form['km_driven'])
        fuel_type = request.form['fuel_type']
        transmission = request.form['transmission']
        brand = request.form['brand']

        # Create input dataframe
        input_df = pd.DataFrame({
            'year': [year],
            'km_driven': [km_driven],
            'fuel_type': [fuel_type],
            'transmission': [transmission],
            'brand': [brand]
        })

        # One-hot encode
        input_df = pd.get_dummies(input_df)

        # Align with training columns
        trained_columns = X_train.columns
        for col in trained_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[trained_columns]

        # Scale numerical columns
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Make predictions
        pred_rf = rf_model.predict(input_df)
        pred_xgb = xgb_model.predict(input_df)
        pred_ensemble = (pred_rf + pred_xgb) / 2

        return render_template('index.html', prediction=round(pred_ensemble[0], 2))

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)