import pandas as pd
import mlflow


#for azure machine learning studio experiment tracking
from azureml.core import Workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


#Set experiment name
mlflow.set_experiment("vhen - test (provided code)")

#Import usefull libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

##own imports##
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

#Start a run
with mlflow.start_run(run_name="test code"):
    df = pd.read_json("./dataset.json", orient="split")
    
    print(df)
    # TO DO: Handle missing data

    pipeline = Pipeline([
            #impute missing data
            ('missing', ColumnTransformer([
                ("impute_wind", SimpleImputer(strategy='mean'), [0]),
                ("impute_direction", SimpleImputer(strategy='most_frequent'), [1]),
                ("impute_total", SimpleImputer(strategy='mean'), [2])
                ]))

            #encode winddirection

            #add polyfeatures

            #scale data

            #add model
        ])


    metrics = [
            ("MAE", mean_absolute_error, []),
            ]

    X = df[["Speed", "Direction"]]
    y = df[["Total"]]
    
    #print(X.mean())
    #print(y)

    number_of_splits = 5

    # TO DO: log your parameters. What parameters are important to log?
    # HINT: You can get access to the transformers in your pipeline using 'pipeline.steps'

    #for train, test in TimeSeriesSplit(number_of_splits).split(X,y):
    #    pipeline.fit(X.iloc[train], y.iloc[train])
    #    predictions = pipeline.predict(X.iloc[test])
    #    truth = y.iloc[test]
    #    
    #    plt.plot(truth.index, truth.values, label="Truth")
    #    plt.plot(truth.index, predictions, label="Predictions")
    #    plt.show()

    #    # calculate and save the metrics for this fold
    #    for name, func, scores in metrics:
    #        score = func(truth, predictions)
    #        scores.append()

    ##Log a summary of the metrics
    #for name, _, scores in metrics:
    #    # NOTe: Here we just log the mean of the scores.
    #    # Are there other summarizations that could be interesting?
    #    mean_score = sum(scores)/number_of_splits
    #    mlflow.log_metric(f"mean_{name}", mean_score)


