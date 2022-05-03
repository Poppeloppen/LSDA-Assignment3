from azureml.core import Workspace, Experiment, Run, VERSION
print("SDK version:", VERSION)

ws = Workspace.from_config()

run = ws.get_run('c3b5b8b3-3dcf-4695-9bf0-fc1c58db965f')
run.cancel()

