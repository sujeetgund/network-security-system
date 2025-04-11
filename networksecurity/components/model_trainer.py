import os
import numpy as np

from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact,
)

from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger
from networksecurity.utils.main_utils import (
    load_object,
    save_object,
    write_yaml,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metrics import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

import mlflow
import dagshub

dagshub.init(repo_owner="sujeetgund", repo_name="network-security-system", mlflow=True)


class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.config = config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e)

    def track_mlflow(
        self, model, classification_metric: ClassificationMetricArtifact
    ) -> None:
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision", classification_metric.precision_score)
            mlflow.log_metric("recall", classification_metric.recall_score)
            mlflow.sklearn.log_model(model, "model")

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> ModelTrainerArtifact:
        """
        Train the model using various classifiers and select the best one based on performance metrics.
        The best model is saved to a specified file path, and a NetworkModel object is created and saved.

        Args:
            X_train (np.ndarray): Training feature set.
            y_train (np.ndarray): Training target variable.
            X_test (np.ndarray): Testing feature set.
            y_test (np.ndarray): Testing target variable.

        Raises:
            NetworkSecurityException: If any error occurs during the training process.

        Returns:
            ModelTrainerArtifact: An object containing the file path of the trained model and performance metrics.
        """
        try:
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['gini', 'entropy', 'log_loss'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['log_loss', 'exponential'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate models
            evaluation_result = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            report = evaluation_result["report"]
            best_model_name = evaluation_result["best_model_name"]
            best_model = evaluation_result["best_model"]
            best_score = evaluation_result["best_score"]
            best_params = evaluation_result["best_params"]

            y_preds_train = best_model.predict(X_train)
            classification_train_metric = get_classification_score(
                y_true=y_train, y_pred=y_preds_train
            )
            # Track mlflow for train metric
            self.track_mlflow(best_model, classification_train_metric)

            y_preds_test = best_model.predict(X_test)
            classification_test_metric = get_classification_score(
                y_true=y_test, y_pred=y_preds_test
            )
            # Track mlflow for test metric
            self.track_mlflow(best_model, classification_test_metric)

            logger.info(f"Best model name: {best_model_name}")
            logger.info(f"Best model params: {best_params}")
            logger.info(f"Best model r2 score: {best_score}")
            logger.info(
                f"Classification train metric: {classification_train_metric.__dict__}"
            )
            logger.info(
                f"Classification test metric: {classification_test_metric.__dict__}"
            )

            # Saving Model Training Report
            logger.info("Saving model training report")
            os.makedirs(
                os.path.dirname(self.config.training_report_file_path), exist_ok=True
            )
            write_yaml(filepath=self.config.training_report_file_path, content=report)

            # Create and Save Network Model
            logger.info("Saving network model")
            preprocessor = load_object(
                filepath=self.data_transformation_artifact.transformed_object_file_path
            )

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)

            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path), exist_ok=True
            )
            save_object(self.config.trained_model_file_path, network_model)
            logger.info(f"Network model saved at {self.config.trained_model_file_path}")

            # Create Model Trainer Artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric,
            )
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e)

    def init_model_trainer(self) -> ModelTrainerArtifact:
        logger.info("Model trainer started")
        try:
            train_file_path = (
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_file_path = (
                self.data_transformation_artifact.transformed_test_file_path
            )

            # loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logger.info("Starting model training")
            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"y_train shape: {y_train.shape}")
            logger.info(f"X_test shape: {X_test.shape}")
            logger.info(f"y_test shape: {y_test.shape}")
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)

            logger.info("Model trainer completed")

            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e)
