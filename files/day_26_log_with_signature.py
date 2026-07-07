"""
Log a model with an MLflow model signature + input example so the
logged model enforces an input schema.

The data and model are synthetic: a DummyClassifier fitted on a small
named two-column frame stands in for a trained model so the logging
call has a real sklearn estimator to persist. The column names
(`amount`, `num_txn`) become the model's input schema. The lab is
about the schema contract, not model quality (§2.5).

Both TODO blocks sit inside the `mlflow.start_run()` context.
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("fraud-signature")

# Synthetic two-feature training frame. The COLUMN NAMES matter — they
# become the input schema that the served model will enforce.
X = pd.DataFrame(
    {
        "amount":  [12.0, 250.0, 7.5, 980.0, 60.0, 410.0],
        "num_txn": [1, 6, 1, 14, 2, 9],
    }
)
y = np.array([0, 1, 0, 1, 0, 1])
model = DummyClassifier(strategy="most_frequent").fit(X, y)
preds = model.predict(X)

with mlflow.start_run():
    # TODO 1: infer the model signature from the training inputs `X` and
    # the model's predictions `preds`, and bind it to `signature`.
    # Use mlflow.models.infer_signature(X, preds).
    signature = infer_signature(X, preds)

    # TODO 2: log the sklearn `model` under the artefact name "model"
    # WITH the inferred `signature` and an `input_example` (pass `X`) so
    # the logged model carries an enforced input schema. Use
    # mlflow.sklearn.log_model(model, name="model", signature=...,
    # input_example=...).
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X,
    )

print("model logged to the fraud-signature experiment")
