# California Housing Regression Analysis

## Overview
This repository contains a linear and polynomial regression analysis of the California housing dataset. The goal was to predict housing prices based on various features.

## Results and Interpretation

### Linear Regression Model
- **Mean Squared Error (MSE)**: 0.5559
- **R-squared (R²)**: 0.5758

### Coefficient Analysis
| Feature     | Coefficient |
|-------------|-------------|
| MedInc      | 0.4487      |
| HouseAge    | 0.0097      |
| AveRooms    | -0.1233     |
| AveBedrms   | 0.7831      |
| Population  | -0.000002   |
| AveOccup    | -0.0035     |
| Latitude    | -0.4198     |
| Longitude   | -0.4337     |

### Intercept
The intercept value is -37.0233.

## Visualization Insights
- **Feature Coefficients**: The bar plot shows which features had the highest and lowest impact on the target.
- **Actual vs. Predicted Values**: Demonstrated how the models performed in comparison to actual data.
- **Residuals**: Histograms indicated how well the model's predictions matched the true values.
