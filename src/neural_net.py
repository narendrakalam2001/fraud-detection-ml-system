from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import RANDOM_STATE
import logging
logger = logging.getLogger(__name__)

# ============================================================
# Train Neural Network separately
# ============================================================

def train_mlp_pipeline(X_train, y_train, preprocessor):

    logger.info("Training Neural Network (MLP)")

    pipe = ImbPipeline([
        ('pre', preprocessor),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(64,32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size = 1024,
            learning_rate='adaptive',
            max_iter=50,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=RANDOM_STATE
        ))
    ])

    pipe.fit(X_train, y_train)

    return pipe