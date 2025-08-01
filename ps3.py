# Weather Forecasting using Machine Learning

# Step 0: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Create a simple weather dataset
data = {
    'MinTemp': [14.1, 13.5, 15.0, 14.2, 13.8, 16.2, 12.5, 11.9, 13.3, 14.8],
    'MaxTemp': [24.3, 25.1, 26.0, 23.5, 22.8, 27.0, 21.5, 20.0, 22.2, 23.9],
    'Rainfall': [0.0, 5.2, 0.0, 0.0, 1.0, 0.0, 15.2, 10.1, 2.0, 0.5],
    'WindSpeed': [10.5, 7.2, 12.3, 9.0, 8.1, 14.5, 5.0, 6.5, 9.3, 10.2],
    'Humidity': [65, 70, 60, 75, 68, 55, 80, 85, 78, 62],
    'TemperatureNextDay': [22.0, 24.0, 25.5, 23.0, 21.5, 26.5, 20.0, 19.0, 21.0, 22.5]
}

df = pd.DataFrame(data)

# Step 2: (Optional) Data visualization
sns.pairplot(df)
plt.suptitle("Weather Data Pairplot", y=1.02)
plt.show()

# Step 3: Prepare the data
X = df.drop('TemperatureNextDay', axis=1)
y = df['TemperatureNextDay']

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("------------------")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Step 7: Make a prediction for new weather data
new_input = pd.DataFrame({
    'MinTemp': [13.0],
    'MaxTemp': [25.0],
    'Rainfall': [0.2],
    'WindSpeed': [9.5],
    'Humidity': [60]
})

# Step 8: Today's avg temperature
today_avg_temp = (new_input['MinTemp'][0] + new_input['MaxTemp'][0]) / 2

# Step 9: Predict tomorrow's temperature
prediction = model.predict(new_input)
predicted_temp = prediction[0]

# Step 10: Print final report
print("\n--- Weather Forecast Report ---")
print(f"Aaj ka average temperature: {today_avg_temp:.2f} °C")
print(f"Kal ka estimated temperature (prediction): {predicted_temp:.2f} °C")

