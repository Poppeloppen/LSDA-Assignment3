################################################################
#                           IMPORTS 
################################################################

import pandas as pd
import mlflow
import math
import numpy as np


#for azure machine learning studio experiment tracking
from azureml.core import Workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


#Set experiment name
#mlflow.set_experiment("vhen - Experiments")
mlflow.set_experiment("TEST - XGBRegressor2")

#Import usefull libraries
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

##own imports##
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor



################################################################
#                       CUSTOM FUNCTIONS 
################################################################

def split_df(df):
    """
    For splitting a single dataframe into features (X) and labels (y)
    
    input:
        df: a dataframe with the columns
            ANM, Non-ANM, Total, Direction, Lead_hours, Source_time, Speed.
    
    Output:
        X: Features to use for model
        y: Labels to use for model

    """
    
    X = df[["Direction", "Speed"]]
    y = df[["Total"]]
       
    return X, y


def direction_vector_encoder(wind_data):
    """
    Takes in the wind_data (numpy array) and transform the string representation of the wind direction
    into a 2D vector representation - each of the dimensions of the vector with its own
    column in the updated data
    """

    #dictionary to map wind direction to degrees
    wind_encoder = {'NNE': 22.5,
                    'NE': 45,
                    'ENE': 67.5,
                    'E': 90,
                    'ESE': 112.5,
                    'SE': 135,
                    'SSE': 157.5,
                    'S': 180,
                    'SSW': 202.5,
                    'SW': 225,
                    'WSW': 247.5,
                    'W': 270,
                    'WNW': 292.5,
                    'NW': 315,
                    'NNW': 337.5,
                    'N': 360}

    #Convert to 'math direction' (degrees) --> as in this provided source: http://colaweb.gmu.edu/dev/clim301/lectures/wind/wind-uv
    md = {k:(270-v if 270-v >=0 else (270-v+360)) for k,v in wind_encoder.items()}

    #Convert the math degrees to radians
    md_rad = {k:math.radians(v) for k,v in md.items()}

    #Add exstra column in the front of the wind_np matrix
    wind_data = np.c_[np.zeros(len(wind_data)), wind_data]

    #Calculate the components of the vector and insert into the np matrix
    for i in range(len(wind_data)):
        u = wind_data[i, 2] * math.cos(md_rad[wind_data[i, 1]])
        v = wind_data[i, 2] * math.sin(md_rad[wind_data[i, 1]])
        wind_data[i, 0] = u
        wind_data[i, 1] = v


    return wind_data #output the updated data




class Debugging(BaseEstimator, TransformerMixin):
    """
    For debbugging a pipeline, e.g. by getting the data or the shape of 
    the data at the current step in the pipeline;
        pipeline.named_steps[<name of key in pipeline>].data
        pipeline.named_steps[<name of key in pipeline>].shape
    """
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        self.shape = X.shape
        self.data = X
        return X      
    
    
class WindDirToVec(BaseEstimator, TransformerMixin):
    """
    For converting the wind direction (string) to a vector representation.
    This is done by calling the 'direction_vector_encoder()' created above
    """
    def __init__(self, run=True):
        self.run = run
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        #Only transform data if run==True
        if self.run:
            X2 = direction_vector_encoder(X)
            return X2
        else:
            return X


class Poly(BaseEstimator, TransformerMixin):
    """
    Make sure the PolynomialFeatures option is optional in the pipeline below,
    And also make sure that one can provide different degrees to the PolynomialFeatures
    library
    """
    def __init__(self, degree, run=True):
        self.run = run
        self.degree = degree
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        #Only transform data if run==True
        if self.run:
            X2 = PolynomialFeatures(degree)
            return X2
        else:
            return X





################################################################
#                       THE PIPELINE 
################################################################
def pipe(model, degree, wind_dir_to_vec=True):
    """
    Creates a pipeline to;
        handle missing values,
        encode string features (wind direction in this case),
        add additional polynomial features
        scale the data with the StandardScaler,
        train a model of the type provided as an argument to the function

    input:
        model: A function object e.g. SVC (if SVC is imported from sklearn),
               this model is the one used for predicting new values
        wind_dir_to_vec: Encode the winddirection as a vector, standard = True
        degree: The degree used by PolynomialFeatures

    output:
        pipeline: A sklearn pipeline object, that takes care of imputing,
        encoding and scaling the data
    """

    #Make pipeline for formatting data
    pipeline = Pipeline([

        #Impute data
        ('missing', ColumnTransformer([
            ("imputeStr", SimpleImputer(strategy='most_frequent'), [0]),
            ("imputeNum", SimpleImputer(strategy='mean'), [1])
        ], remainder='passthrough')),



        #Encode wind direction (ONE HOT)
        #('encode', ColumnTransformer([
        #    ("encodeStr", OneHotEncoder(sparse=False), [0])], remainder='passthrough')),

        #Encode wind direction (2D VEC)
        ('encode', WindDirToVec(wind_dir_to_vec)),

        #Make sure you can see the data before additional features are added
        ('debug1', Debugging()),

        #Add poly-features
        ('poly_features', Poly(degree=degree, run=False)),

        #Make sure you can see the data before it gets scaled
        ('debug2', Debugging()),

        #Scale data
        ('std_scaler', StandardScaler()),

        #Add regression model
        ('model', model)
    ])

    return pipeline



################################################################
#                       THE MLFLOW RUN 
################################################################

params = {"number_of_splits": [2,5,10],
        "learning_rate": [0.0001, 0.001, 0.01, 0.2, 0.3],
        "n_estimators": [10, 50, 100, 200, 500],
        "max_depth": [1,2,3,4,5]
        }

for splits in params["number_of_splits"]:
    for learning_rate in params["learning_rate"]:
        for estimators in params["n_estimators"]:
            for depth in params["max_depth"]:
                print("# of splits: ", splits)
                print("learning_rate: ", learning_rate)
                print("# estimators: ", estimators)
                print("max depth: ", depth)
                print('#' * 90)

                #Start a run
                with mlflow.start_run(run_name="XGBRegressor"):
                    df = pd.read_json("./dataset.json", orient="split")
                
                    #Only keep rows where there are no missing values along the "Direction" column
                    # This corresponds to all the rows that have no missing values along all columns
                    complete_data = df[~df["Direction"].isnull()]
                    
                
                    #TO DO: Currently the only metric is MAE. You should add more. What other metrics could you use? why?
                    metrics = [
                            ("MAE", mean_absolute_error, []),
                            ("MSE", mean_squared_error, []),
                            ("r2", r2_score, []) 
                            ]
                
                    X, y = split_df(complete_data)
                    
                    #######################
                    # Hyperparameters
                    #######################
                    parameters = {"number_of_splits": splits,    #To do in crossvali
                            "learning_rate": learning_rate, #DOES THIS MAKES SENSE WHEN NOT DOING LINREG?!
                            "n_estimators": estimators,
                            "max_depth": depth}
                
                    mlflow.log_params(parameters)
                    # TO DO: log your parameters. What parameters are important to log?
                    # HINT: You can get access to the transformers in your pipeline using 'pipeline.steps'
                
                    model = XGBRegressor(n_estimators=estimators,
                                            max_depth = depth,
                                            learning_rate = learning_rate)
                
                
                    for train, test in TimeSeriesSplit(splits).split(X,y):
                        pipeline = pipe(model, degree = 2)
                        pipeline.fit(X.iloc[train], y.iloc[train].values.ravel())
                        predictions = pipeline.predict(X.iloc[test])
                        truth = y.iloc[test]
                        
                        #fig = plt.figure()
                        #ax = fig.add_axes([0.2,0.2,0.7,0.7])
                        #ax.plot(truth.index, truth.values, label="Truth")
                        #ax.plot(truth.index, predictions, label="Predictions")
                        #fig.legend()
                        #fig.autofmt_xdate(rotation=45)
                        #plt.show()
                
                        # calculate and save the metrics for this fold
                        for name, func, scores in metrics:
                            score = func(truth, predictions)
                            scores.append(score)
                
                    #Log a summary of the metrics
                    for name, _, scores in metrics:
                        # NOTe: Here we just log the mean of the scores.
                        # Are there other summarizations that could be interesting?
                        mean_score = sum(scores)/splits
                        mlflow.log_metric(f"mean_{name}", mean_score)
                

                        # Make sure to track the hyper-parameters, eg. degrees in PolynomialFeatures
                        #mlflow.log_params()
                
                
    

















#Start a run
#with mlflow.start_run(run_name="RandomForestRegressor"):
#    df = pd.read_json("./dataset.json", orient="split")
#
#    #Only keep rows where there are no missing values along the "Direction" column
#    # This corresponds to all the rows that have no missing values along all columns
#    complete_data = df[~df["Direction"].isnull()]
#    
#
#    #TO DO: Currently the only metric is MAE. You should add more. What other metrics could you use? why?
#    metrics = [
#            ("MAE", mean_absolute_error, []),
#            ("MSE", mean_squared_error, []),
#            ("r2", r2_score, []) 
#            ]
#
#    X, y = split_df(complete_data)
#    
#    #######################
#    # Hyperparameters
#    #######################
#    params = {"number_of_splits": 5,    #To do in crossvali
#            "number_of_poly_degree": 1, #DOES THIS MAKES SENSE WHEN NOT DOING LINREG?!
#            "n_estimators": 500,
#            "max_depth": 5}
#
#    mlflow.log_params(params)
#    
#    print("# of splits: ", params["number_of_splits"])
#    print("# of degree: ", params["number_of_poly_degree"])
#    print("# estimators: ", params["n_estimators"])
#    print("max depth: ", params["max_depth"])
#    # TO DO: log your parameters. What parameters are important to log?
#    # HINT: You can get access to the transformers in your pipeline using 'pipeline.steps'
#
#    model = RandomForestRegressor(n_estimators=params['n_estimators'],
#                                    max_depth=params['max_depth'])
#
#
#    for train, test in TimeSeriesSplit(params["number_of_splits"]).split(X,y):
#        pipeline = pipe(model, params["number_of_poly_degree"])
#        pipeline.fit(X.iloc[train], y.iloc[train].values.ravel())
#        predictions = pipeline.predict(X.iloc[test])
#        truth = y.iloc[test]
#        
#        #fig = plt.figure()
#        #ax = fig.add_axes([0.2,0.2,0.7,0.7])
#        #ax.plot(truth.index, truth.values, label="Truth")
#        #ax.plot(truth.index, predictions, label="Predictions")
#        #fig.legend()
#        #fig.autofmt_xdate(rotation=45)
#        #plt.show()
#
#        # calculate and save the metrics for this fold
#        for name, func, scores in metrics:
#            score = func(truth, predictions)
#            scores.append(score)
#
#    #Log a summary of the metrics
#    for name, _, scores in metrics:
#        # NOTe: Here we just log the mean of the scores.
#        # Are there other summarizations that could be interesting?
#        mean_score = sum(scores)/params["number_of_splits"]
#        mlflow.log_metric(f"mean_{name}", mean_score)
#
#
#        # Make sure to track the hyper-parameters, eg. degrees in PolynomialFeatures
#        #mlflow.log_params()
#
#
