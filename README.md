# california_housing_regression

Results and Interpretation of the Regression Analysis
Linear Regression Model:
Mean Squared Error (MSE): The MSE for the linear model was calculated to be 0.5559. This value provides insight into the average squared difference between the predicted and actual prices. A lower MSE indicates that the model's predictions are closer to the true values, suggesting that the model performs reasonably well but might still benefit from further tuning or inclusion of additional features.
R-squared (R²): The R² value for the linear model was 0.5758, which suggests that 57.58% of the variance in the target variable (price) is explained by the features in the dataset. This indicates that while the model can capture some of the underlying patterns, there is still a substantial portion of variance unexplained by the independent variables.
Coefficient Analysis:

The coefficients from the linear model reveal the influence each feature has on the target variable. For instance:
MedInc (Median Income): Coefficient of 0.4487 — Indicates that an increase in median income is associated with a significant positive change in predicted price.
HouseAge: Coefficient of 0.0097 — Shows that as the age of the house increases, the price slightly increases.
AveRooms (Average Rooms per House): Coefficient of -0.1233 — A negative value suggesting that, on average, an increase in rooms per house corresponds to a decrease in price.
AveBedrms (Average Bedrooms per House): Coefficient of 0.7831 — Indicates a positive correlation with house prices.
Population: Coefficient of -0.000002 — A very small negative coefficient implying minimal impact on price.
AveOccup (Average Occupancy per House): Coefficient of -0.0035 — Slight negative impact on price.
Latitude: Coefficient of -0.4198 — A negative value, suggesting that houses in northern latitudes tend to have lower prices.
Longitude: Coefficient of -0.4337 — Indicates that houses further west tend to be cheaper.
Intercept:

The intercept value was -37.0233, representing the predicted value of the target when all input features are zero. This value has limited practical interpretation, as it does not reflect realistic conditions for the input data.
Polynomial Regression Model (Degree 2):
Polynomial Model Prediction: By transforming the input features into a polynomial space, I fitted a more complex model capable of capturing non-linear relationships within the data. This transformation enabled a more nuanced prediction and was visually compared against the actual prices and the linear model's predictions to assess improvements in accuracy and fit.
Residual Analysis:

Linear Model Residuals: The residuals for the linear regression model showed a pattern that suggests underfitting, where errors are more dispersed and do not follow a clear pattern, indicating potential areas where the model could be improved.
Polynomial Model Residuals: The residuals from the polynomial model were more evenly distributed and clustered around zero, indicating a better fit compared to the linear model. This suggests that the polynomial regression was able to capture more complex patterns in the data, leading to fewer systematic errors.
Visualization Insights:
Actual vs. Predicted Values Plot: The plot clearly showed that the polynomial model closely followed the actual prices, indicating better prediction accuracy, while the linear model exhibited greater deviations from the actual values. This visual representation emphasized the advantages of using polynomial regression for this dataset.
Residual Plots: The histograms of residuals for both models showed that the polynomial model had more consistent residuals centered around zero compared to the linear model, indicating a better model fit.
Feature Coefficients Visualization:
The bar plot depicting the feature coefficients for the linear model helped identify which features had the most significant impact on the prediction. Notably, features like MedInc and AveBedrms had higher positive coefficients, suggesting a strong positive relationship with price. In contrast, Latitude and Longitude had negative coefficients, indicating an inverse relationship with price.
