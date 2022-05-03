"""Look through all the mlflow runs on azure and save the best model"""
import mlflow

#for azure machine learning studio experiment tracking
from azureml.core import Workspace, Experiment
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

print("EEEEEYYYYYY, U managed to call the 'best_model.py' script. WAY TO GO!!!")

print(ws.experiments)




for exp in ws.experiments:
    experiment = Experiment(ws, exp)
   

    for run in experiment.get_runs():
        print(run.get_metrics()["mean_MSE"])



