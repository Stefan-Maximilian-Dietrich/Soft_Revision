import openml
from sklearn import impute, tree, pipeline

# Define a scikit-learn classifier or pipeline
clf = pipeline.Pipeline(
    steps=[
        ('imputer', impute.SimpleImputer()),
        ('estimator', tree.DecisionTreeClassifier())
    ]
)
# Download the OpenML task for the pendigits dataset with 10-fold
# cross-validation.
task = openml.tasks.get_task(32)
# Run the scikit-learn model on the task.
run = openml.runs.run_model_on_task(clf, task)
# Publish the experiment on OpenML (optional, requires an API key.
# You can get your own API key by signing up to OpenML.org)
run.publish()
print(f'View the run online: {run.openml_url}')
