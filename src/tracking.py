"""
MLflow experiment tracking and model registry utilities.

This module provides functions for logging experiments, registering models,
and managing model lifecycle stages using MLflow. It handles model signatures,
metrics logging, and model promotion to production stages.
"""
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
from skore import EstimatorReport

from src.globals import logger


def log_and_register_model_with_mlflow(final_model, test_df, cfg, params):
    """
    Log and register a trained model with MLflow tracking and model registry.

    This function logs the model, parameters, and metrics to MLflow, infers the model
    signature, and registers the model in the MLflow model registry for deployment.

    Args:
        final_model: Trained scikit-learn model to be logged and registered
        test_df: Test dataset used for model evaluation and signature inference
        cfg: Configuration dictionary containing MLflow settings, paths, and model names
        params: Dictionary of hyperparameters and training parameters to log

    Returns:
        Tuple[mlflow.entities.model_registry.ModelVersion, str]: 
            A tuple containing the registered model details and the MLflow run ID
    """
    # to remove warning of representing np.nan
    integer_columns = test_df.select_dtypes(include=["int"]).columns
    test_df[integer_columns] = test_df[integer_columns].astype(float)
    X = test_df.drop(columns=cfg["dataset"]["target_col"])
    y = test_df[cfg["dataset"]["target_col"]]
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])
    with mlflow.start_run():
        mlflow.autolog()
        run_id = mlflow.active_run().info.run_id

        # Infer the model signature
        train_preds = final_model.predict(X)
        signature = infer_signature(X, train_preds)

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path=cfg["paths"]["models_parent_dir"],
            registered_model_name=cfg["names"]["model_name"],
            signature=signature,
            input_example=X.iloc[0:1],
        )

        # Log parameters and metrics
        mlflow.log_params(params)
        final_report = EstimatorReport(final_model, X_test=X, y_test=y)
        metrics = {
            "accuracy": final_report.metrics.accuracy(),
            "precision": final_report.metrics.precision(),
            "recall": final_report.metrics.recall(),
            "roc_auc": final_report.metrics.roc_auc(),
        }
        mlflow.log_metrics(reformat_metrics(metrics))

        # Register the model
        model_uri = f"runs:/{run_id}/{cfg['paths']['models_parent_dir']}"
        model_details = mlflow.register_model(model_uri=model_uri, name=cfg["names"]["model_name"])

        logger.error(model_details)
        logger.info("Model registered successfully!!")

        return model_details, run_id


def reformat_metrics(metrics):
    """
    Reformat nested metrics dictionary for MLflow logging.
    
    Flattens nested dictionaries by combining keys with underscores,
    making them suitable for MLflow's flat metrics structure.
    
    Args:
        metrics (dict): Dictionary of metrics that may contain nested dictionaries
        
    Returns:
        dict: Flattened dictionary with reformatted metric names
    """
    reformatted_metrics = dict()
    for metric, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                reformatted_metrics[f"{metric}_{k}"] = v
        else:
            reformatted_metrics[metric] = value
    return reformatted_metrics


def move_model_to_prod(client: mlflow.client.MlflowClient, model_details) -> None:
    """
    Transition a model version to production stage in MLflow model registry.
    
    Moves the specified model version to the "production" stage and adds
    a production tag for easy identification.
    
    Args:
        client (mlflow.client.MlflowClient): MLflow client for model registry operations
        model_details: Model version details from MLflow registration
        
    Returns:
        None
    """
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="production",
    )
    # add tag that the model is in production
    client.set_model_version_tag(
        name=model_details.name,
        version=model_details.version,
        key="production",
        value="true",
    )
    logger.info("Model transitioned to prod stage")
