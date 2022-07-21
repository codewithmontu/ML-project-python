import pandas as pd
import numpy as np
housing = pd.read_csv("data.csv")
print(housing.head(15))
print(housing.info())
print(housing['CHAS'].value_counts())
print(housing.describe())
print("\n")

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print("Lets Check the Train & Test Split ratio of CHAS : \n")
print(strat_test_set['CHAS'].value_counts())
print(strat_train_set['CHAS'].value_counts())
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()

from sklearn.impute import SimpleImputer
median = housing["RM"].median() 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)
#print(housing_num_tr.shape)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
print("\n")
print("Some Predicted Values of Train : \n",model.predict(prepared_data))
print("Some Real Values of Train : \n",list(some_labels))
print("\n")
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,classification_report
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
#print(rmse_scores)
print("\n")
def print_scores(scores):
    print("RMSE Scores of Train : ", scores)
    print("\n")
    print("Mean of Train : ", scores.mean())
    print("\n")
    print("Standard deviation of Train : ", scores.std())
print_scores(rmse_scores)
print("\n")
print("-----------------------------------------------------------------")
print("\n")
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
housing_tr = pd.DataFrame(X_test_prepared, columns=X_test.columns)
#print(housing_tr.head(30))
final_predictions = model.predict(X_test_prepared)
print("Final Predicted Values  of Test : \n\n",final_predictions)
print("\n")
print("Final Real Values of Test : \n\n",list(Y_test))
print("\n")

final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))
print("Final Root Mean Square Error : ",final_rmse)
print("\n")
#print(confusion_matrix(Y_test,final_predictions))
#print(accuracy_score(Y_test,final_predictions))
#print(precision_score(Y_test,final_predictions))
#print(recall_score(Y_test,final_predictions))
#print(f1_score(Y_test,final_predictions))

