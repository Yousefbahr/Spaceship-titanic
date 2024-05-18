import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from warnings import simplefilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 50)
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

data["total amount"] = data["RoomService"]+  data["FoodCourt"] +  data['ShoppingMall'] +  data['Spa'] + data['VRDeck']
test_data["total amount"] = test_data["RoomService"]+  test_data["FoodCourt"] +  test_data['ShoppingMall'] +  test_data['Spa'] + test_data['VRDeck']

train_data, val_data = train_test_split(data, test_size=0.2)

input_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt',
              'ShoppingMall', 'Spa', 'VRDeck', 'total amount', "Cabin"]
# input_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP', 'total amount']
target_col = ["Transported"]

train_input = train_data[input_cols]
train_target = train_data[target_col]

val_input = val_data[input_cols]
val_target = val_data[target_col]

test_input = test_data[input_cols]

categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP',"Cabin"]
numerical_cols = ["Age", "RoomService", "FoodCourt", 'ShoppingMall', 'Spa', 'VRDeck', "total amount"]
# numerical_cols = ["Age", "total amount"]
# CLEAN DATA

# Missing Values
imp = SimpleImputer(strategy="most_frequent")
imp.fit(data[categorical_cols])


imp2 = SimpleImputer(strategy="median")
imp2.fit(data[numerical_cols])

train_input[categorical_cols] = imp.transform(train_input[categorical_cols])
val_input[categorical_cols] = imp.transform(val_input[categorical_cols])
test_input[categorical_cols] = imp.transform(test_data[categorical_cols])

train_input[numerical_cols] = imp2.transform(train_input[numerical_cols])
val_input[numerical_cols] = imp2.transform(val_input[numerical_cols])
test_input[numerical_cols] = imp2.transform(test_input[numerical_cols])

# Scale numerical values
scaler = MinMaxScaler()
scaler.fit(data[numerical_cols])

train_input[numerical_cols] = scaler.transform(train_input[numerical_cols])
val_input[numerical_cols] = scaler.transform(val_input[numerical_cols])
test_input[numerical_cols] = scaler.transform(test_input[numerical_cols])

# Encode Categorical values
encoder = OneHotEncoder(sparse_output =False, handle_unknown="ignore")
encoder.fit(data[categorical_cols])

encoded_categ = list(encoder.get_feature_names_out())

train_input[encoded_categ] = encoder.transform(train_input[categorical_cols])
val_input[encoded_categ] = encoder.transform(val_input[categorical_cols])
test_input[encoded_categ] = encoder.transform(test_input[categorical_cols])

# WE COULD MAKE A NEW FEATURE OF TOTAL AMOUNT SPENT ON TITANIC
# I AM THINKING OF NOT SCALING AGE FROM 0 TO 1

dumb_model = pd.Series(np.full((val_data.shape[0]), "False")).astype(bool)


model = RandomForestClassifier(n_estimators=250,n_jobs=-1, random_state=42,max_depth=200)
model.fit(train_input[encoded_categ+ numerical_cols], train_target)

print(f"model accuracy on training data: {accuracy_score(train_target, model.predict(train_input[encoded_categ+numerical_cols]))}")
print(f"model accuracy on validating data: {accuracy_score(val_target, model.predict(val_input[encoded_categ+numerical_cols]))}")


print(f"dumb model accuracy on val data: {accuracy_score(val_target, dumb_model)}")

# # My answer
# test_pred = model.predict(test_input[ encoded_categ + numerical_cols])
# with open("answer.csv","w") as file:
#     file.write(f"PassengerId,Transported\n")
#     for ID, answer in zip(test_data["PassengerId"], test_pred):
#         file.write(f"{ID},{answer}\n")