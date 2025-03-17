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

    # Data Cleaning Step
    cleaned_data = clean_data(raw_data)

    # Feature Engineering Step
    X_raw, y_raw, dates = feature_engineering_step(
        cleaned_data
    )

    # Data Splitting 
    X_train, X_val, X_test, y_train, y_val, y_test = data_splitter_step(X_scaled=X_raw, y_scaled=y_raw, dates=dates)

    # Model Training
    model = model_training_step(X_train, y_train, X_val, y_val)

    # Model Evaluation
    evaluator = model_evaluation_step(model, X_train, y_train, X_val, y_val, X_test, y_test)

    return evaluator

if __name__=="__main__":
    run = ml_pipeline()
