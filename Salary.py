# Step 1: import modules
import pandas as pd
#Step 2: import data
Salary=pd.read_csv("https://github.com/YBI-Foundation/Dataset/raw/main/Salary%20Data.csv")
Salary.head()
Salary.info()
Salary.describe()
Salary.columns
Salary.shape
#Step 3: Define x and y
Salary.columns
y=Salary['Salary']
X=Salary[['Experience Years']]
#Step 4: train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state=2529)
X_train.shape,X_test.shape,y_train.shape,y_test.shape
#Step 5: select the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#Step 6: train the model
model.fit(X_train,y_train)