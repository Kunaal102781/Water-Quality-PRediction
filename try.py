import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset
file_path = "C:\Water Quality PRediction\water_dataX.csv"
df = pd.read_csv(file_path, encoding='unicode_escape')

# Drop irrelevant columns (if any)
df.drop(columns=['year'], inplace=True)

# Handling missing values
df.replace('NAN', np.nan, inplace=True)
df.fillna(df.median(), inplace=True)  # Fill missing values with median

# Convert categorical variables to numerical using one-hot encoding (if needed)
df = pd.get_dummies(df)

# Split data into features and target variable
X = df.drop(columns=['Temp'])  # Features
y = df['Temp']  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)

print("Model Evaluation on Training Data:")
evaluate_model(model, X_train, y_train)
print("\nModel Evaluation on Test Data:")
evaluate_model(model, X_test, y_test)

# Example prediction
new_data = pd.DataFrame([[...]], columns=X.columns)  # Replace ... with new input data
new_data_preprocessed = pd.get_dummies(new_data)  # Preprocess the new data if needed
prediction = model.predict(new_data_preprocessed)
print("Predicted Temperature:", prediction)
