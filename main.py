import os
from typing import Dict

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.dataset import load_dataset
from src.evaluate import evaluate
from src.features import create_modular_pipeline
from src.globals import logger
from src.preprocessor import Preprocessor
from src.saver import Saver
from src.tracking import log_and_register_model_with_mlflow, move_model_to_prod
from src.training import train_RandomizedSearchCV


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ############################## 1. Load Training Data ##############################
    logger.info("1. Load Training Data")
    # training data path
    train_path = os.path.join(
        cfg["paths"]["data"]["raw_data"],
        f"{cfg['names']['train_data']}.csv"
    )
    train_df = load_dataset(train_path)
    print(f"Training data shape: {train_df.shape}")

    ############################## 2. apply Feature engineering ##############################
    logger.info("2. apply Feature engineering")
    feature_pipeline = create_modular_pipeline()

    feature_engineered_train = feature_pipeline.fit_transform(train_df)
    print(f"Feature engineered training data shape: {feature_engineered_train.shape}")
    Saver.save_dataset_csv(
        dataset=feature_engineered_train,
        file_path=cfg["paths"]["data"]["interim_data"],
        file_name=f"{cfg['names']['train_data']}.csv"
,
    )

    ############################## 3. split data ##############################
    logger.info("3. split data")
    X_train, X_val, y_train, y_val = train_test_split(
        feature_engineered_train.drop(columns=cfg["dataset"]["target_col"]),
        feature_engineered_train[cfg["dataset"]["target_col"]],
        test_size=cfg["dataset"]["test_size"],
        random_state=cfg["dataset"]["random_state"],
    )

    # ######################## 4. Preprocessing ##############################
    logger.info("4. Preprocessing")
    pipeline_config: Dict = OmegaConf.to_container(cfg["pipeline_config"], resolve=True)
    preprocessor_pipeline = Preprocessor(pipeline_config)

    X_processed_train = preprocessor_pipeline.fit_transform(X_train)
    y_processed_train = y_train.values
    print(f"Processed training data shape: {X_processed_train.shape}")
    Saver.save_dataset_npy(
        dataset=X_processed_train,
        file_path=cfg["paths"]["data"]["processed_data"],
        file_name=cfg["names"]["train_data"],
    )
    Saver.save_dataset_npy(
        dataset=y_processed_train,
        file_path=cfg["paths"]["data"]["processed_data"],
        file_name=cfg["names"]["target_name"],
    )

    ########################## 5. Train Model ##############################
    logger.info("5. Train Model")
    model = RandomForestClassifier(
        **OmegaConf.to_container(cfg["hyperparameters"]["random_forest"], resolve=True),
    )

    model, best_params = train_RandomizedSearchCV(model, cfg, X_processed_train, y_processed_train)

    ########################## 6. save model ##############################
    logger.info("6. save model")
    full_pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_pipeline),
            ("preprocessor", preprocessor_pipeline.get_pipeline()),
            ("model", model),
        ]
    )
    Saver.save_model(
        full_pipeline,
        model_path= cfg["paths"]["models_parent_dir"],
        model_name=cfg['names']['model_name'])
    
    ########################### 7. experiment tracking ##############################
    logger.info("7. experiment tracking")
    if cfg["mlflow"]['is_tracking_enabled']:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        client = mlflow.tracking.MlflowClient()
        model_details, run_id = log_and_register_model_with_mlflow(
            final_model=full_pipeline,
            test_df=train_df,
            cfg=cfg,
            params=best_params,
        )

        move_model_to_prod(
            client=client,
            model_details=model_details,
        )

    ############################ 8. evaluate Model ##############################
    logger.info("8. evaluate Model")
    X_processed_val = preprocessor_pipeline.transform(X_val)
    y_processed_val = y_val.values    
    Saver.save_dataset_npy(
        dataset=X_processed_val,
        file_path=cfg["paths"]["data"]["processed_data"],
        file_name=cfg["names"]["val_data"],
    )
    Saver.save_dataset_npy(
        dataset=y_processed_val,
        file_path=cfg["paths"]["data"]["processed_data"],
        file_name=cfg["names"]["target_name"],
    )
    evaluate(cfg=cfg, final_model=model, X_test=X_processed_val, y_test=y_processed_val)

if __name__ == "__main__":
    main()
