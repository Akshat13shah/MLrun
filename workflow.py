from kfp import dsl
import mlrun
from mlrun.model import HyperParamOptions

@dsl.pipeline(
    name="Breast Cancer ML Pipeline",
    description="A pipeline to preprocess data and train a breast cancer classifier with hyperparameter tuning."
)
def cancer_pipeline():
    project = mlrun.get_current_project()

    # Step 1: Data preparation
    data_prep = project.run_function(
        "data-prep",
        name="prepare-data",
        handler="fetch_data",
        outputs=["data"]
    )

    # Step 2: Model training with hyperparameter tuning
    train = project.run_function(
        "train-function",
        handler="train_model",
        name="train-model",
        hyperparams={
            "n_estimators": [10, 100, 200],
            "max_depth": [2, 5, 10]
        },
        hyper_param_options=HyperParamOptions(
            strategy="grid",
            selector="max.accuracy"
        ),
        inputs={"dataset": data_prep.outputs["data"]}
    )
