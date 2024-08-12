# Step 1: import modules
import pandas as pd

#Step 2: import data
dbs = pd.read_csv("https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes.csv")
dbs.head()
dbs.info()
dbs.describe()
dbs.columns
dbs.shape

#Step 3: Define x and y
dbs.columns
y=dbs['diabetes']
X=dbs[['pregnancies', 'glucose', 'diastolic', 'triceps', 'insulin', 'bmi',
       'dpf', 'age']]

#Step 4: train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

#Step 5: select the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Step 6: train the model
model.fit(X_train,y_train)
model.intercept_    #intercept 
model.coef_         #slope 

#Step 7: Predict
y_pred = model.predict(X_test)

#Step 8: Mean squared error and r2_score
from sklearn.metrics import mean_squared_error, r2_score # import appropriate metrics for regression

# Evaluate model performance using regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}") 