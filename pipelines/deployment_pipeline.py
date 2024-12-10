from zenml import pipeline

from pipelines.training_pipeline import ml_pipeline
from steps.dynamic_importer import dynamic_importer
from steps.prediction_service_loader import prediction_service_loader
from steps.predictor import predictor
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

# from steps.dynamic_importer import dynamic_importer
# from steps.prediction_service_loader import prediction_service_loader
# from steps.predictor import predictor


@pipeline
def continuous_deployment_pipeline():
    """
    This pipeline is responsible for continuously deploying trained models. It first runs the ml_pipeline to train a model,
    then uses the mlflow_model_deployer_step to deploy the trained model. The deployment decision is based on the model's performance.

    :return: None
    """
    trained_model = ml_pipeline()

    mlflow_model_deployer_step(
        workers=3,  # Trained model is deployed with 3 workers.
        deploy_decision=True,  # Deployment decision is based on the model's performance.
        model=trained_model,
    )
  


@pipeline
def inference_pipeline(enable_cache=True):
    """Run a batch inference job with data loaded from an API."""
    # Load batch data for inference
    batch_data = dynamic_importer()

    # Load the deployed model service
    model_deployment_service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        step_name="mlflow_model_deployer_step",
    )

    # Run predictions on the batch data
    predictor(service=model_deployment_service, input_data=batch_data)