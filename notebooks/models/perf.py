from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    fbeta_score,
    f1_score,
    confusion_matrix,
)
import os

# define a function to evaluate the model performance, including accuracy, balanced accuracy, f0.5 score, and save them to a csv file


def evaluate_model(model, X_test, y_test, target_name, model_name, feature_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f05_score = fbeta_score(y_test, y_pred, beta=0.5)
    f2_score = fbeta_score(y_test, y_pred, beta=2)
    f1 = f1_score(y_test, y_pred)
    specificity = confusion_matrix(y_test, y_pred)[0, 0] / sum(y_test == 0)
    sensitivity = confusion_matrix(y_test, y_pred)[1, 1] / sum(y_test == 1)

    # save to csv if the file already exists, delete the file otherwise and create a new one
    if os.path.exists(f"result/perf_{target_name}_{model_name}_{feature_name}.csv"):
        # delete the file
        os.remove(f"result/perf_{target_name}_{model_name}_{feature_name}.csv")
    with open(f"result/perf_{target_name}_{model_name}_{feature_name}.csv", "w") as f:
        f.write(
            "target_name,model_name,feature_name,accuracy,balanced_accuracy,f05_score,f2_score,f1,specificity,sensitivity\n"
        )
        f.write(
            f"{target_name},{model_name},{feature_name},{accuracy},{balanced_accuracy},{f05_score},{f1},{f2_score},{specificity},{sensitivity}\n"
        )
