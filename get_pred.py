"""Should contain the pipeline and the best performing model, should be able to get prediction"""

#for azure machine learning studio experiment tracking
from azureml.core import Workspace
ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

print("THIS IS JUST FOR TESTING, you are in 'get_pred.py')
