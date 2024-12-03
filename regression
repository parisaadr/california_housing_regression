from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
housing = fetch_california_housing(as_frame=True)
data = housing.data
data['PRICE'] = housing.target

X = data.drop('PRICE', axis=1)
y = data['PRICE']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_model.predict(X_test)

# Evaluate the linear model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Print coefficients for interpretation
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": linear_model.coef_})
print(coefficients)

# Print intercept
print("Intercept:", linear_model.intercept_)

# Polynomial Regression Model (degree 2 for example)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

# Predict with the polynomial model
y_poly_pred = poly_model.predict(X_poly_test)

# Plotting Actual vs. Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Linear Model')
plt.scatter(y_test, y_poly_pred, color='red', alpha=0.5, label='Polynomial Model')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='black', linewidth=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()

# Plotting Residuals
residuals = y_test - y_pred
poly_residuals = y_test - y_poly_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, color='blue')
plt.title('Residuals for Linear Regression')

plt.subplot(1, 2, 2)
sns.histplot(poly_residuals, kde=True, color='red')
plt.title('Residuals for Polynomial Regression')

plt.show()

# Feature Analysis - Visualizing Coefficients for the Linear Model
plt.figure(figsize=(12, 6))
sns.barplot(x=coefficients.index, y=coefficients['Coefficient'], color='teal')
plt.title('Feature Coefficients for Linear Regression Model')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=90)
plt.show()
