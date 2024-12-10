from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.feature_engineering import feature_engineering_step
from steps.data_splitter import data_splitter_step
from steps.model_training import model_training_step
from steps.model_evaluation import model_evaluation_step
from zenml import Model, pipeline

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="bitcoin_price_predictor"
    ),
)
def ml_pipeline():

    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = ingest_data()

    cleaned_data = clean_data(raw_data)

    transformed_data, X_scaled, y_scaled, scaler_y = feature_engineering_step(
        cleaned_data
    )

    X_train, X_test, y_train, y_test = data_splitter_step(X_scaled=X_scaled, y_scaled=y_scaled)

    model = model_training_step(X_train, y_train)

    evaluator = model_evaluation_step(model, X_test=X_test, y_test=y_test, scaler_y= scaler_y)

    return evaluator

if __name__=="__main__":
    run = ml_pipeline()
