from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logger


class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e)

    def predict(self, X):
        logger.info("Starting prediction process")
        try:
            logger.info("Transforming input data")
            X_transformed = self.preprocessor.transform(X)

            logger.info("Making predictions")
            y_pred = self.model.predict(X_transformed)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e)
