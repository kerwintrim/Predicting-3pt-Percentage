#Libraries that I am using to predict 3pt %
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#This resource was used in identifying how to read a csv file: https://www.w3schools.com/python/pandas/pandas_csv.asp 
#fas.csv file given, df = data frame
file_path = '/Users/kerwin/Desktop/FAS/fas_2024.csv' 
df = pd.read_csv(file_path)
#df.head() is used to test if the object has the right data in it
print(df.head())

#Learned how to create the model below using this link: https://stackoverflow.com/questions/38285345/how-to-predict-stock-price-for-the-next-day-with-python
# X = independent variables, Y = dependent variable
X = df[['ft_pct_oct_nov','three_non_cnr_pct_oct_nov','three_cnr_pct_oct_nov']]
Y = df['three_pct_season']

#Chose random state as 42 based on popularity, chose test size as .2 since training is the harder task and there's a good chance of accomplishing the task with just 80% of data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Linear Regression model being trained
model= LinearRegression()
model.fit(X_train, Y_train)

#Predictions
Y_pred = model.predict(X_test)

# shows avg squared difference between predicted and actual values
MSE = mean_squared_error(Y_test, Y_pred)
print(f'Mean Squared Error: {MSE}')

#Visualizations for each variable used. This link ws used as a reference: https://www.w3schools.com/python/matplotlib_scatter.asp 
plt.scatter(X_test['ft_pct_oct_nov'], Y_test, color='black', label='Actual')
plt.scatter(X_test['ft_pct_oct_nov'], Y_pred, color='blue', label='Predicted')
plt.xlabel('ft_pct_oct_nov')
plt.ylabel('threePointPercentage')
plt.legend()
plt.show()

plt.scatter(X_test['three_non_cnr_pct_oct_nov'], Y_test, color='black', label='Actual')
plt.scatter(X_test['three_non_cnr_pct_oct_nov'], Y_pred, color='blue', label='Predicted')
plt.xlabel('three_non_cnr_pct_oct_nov')
plt.ylabel('threePointPercentage')
plt.legend()
plt.show()

plt.scatter(X_test['three_cnr_pct_oct_nov'], Y_test, color='black', label='Actual')
plt.scatter(X_test['three_cnr_pct_oct_nov'], Y_pred, color='blue', label='Predicted')
plt.xlabel('three_cnr_pct_oct_nov')
plt.ylabel('threePointPercentage')
plt.legend()
plt.show()

end_of_season_predictions = model.predict(df[['ft_pct_oct_nov','three_non_cnr_pct_oct_nov','three_cnr_pct_oct_nov']])

#Saves the new DataFrame to a new CSV file
df['three_pct_end_of_season'] = end_of_season_predictions
df.to_csv('/Users/kerwin/Desktop/FAS/eos_predictions.csv', index=False)
