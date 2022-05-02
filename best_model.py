"""Look through all the mlflow runs on azure and save the best model"""

#for azure machine learning studio experiment tracking
from azureml.core import Workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

print("EEEEEYYYYYY, U managed to call the 'best_model.py' script. WAY TO GO!!!")

