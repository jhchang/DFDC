import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("/my-experiment")
with mlflow.start_run():
    mlflow.log_param("a", 1)
    mlflow.log_metric("b", 2)
    mlflow.log_artifact(__file__)