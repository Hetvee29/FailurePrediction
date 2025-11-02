import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlflow.models.signature import infer_signature
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
print("Starting MLflow tracking...")

df = pd.read_csv("data_preprocessed.csv")
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

mlflow.set_tracking_uri("file:./mlruns")
mlflow.autolog(disable=True)

# Binary classification experiment
mlflow.set_experiment("failure-prediction-binary")
X_bin = df.drop(columns=["Machine_failure", "Type_of_failure"], errors="ignore")
y_bin = df["Machine_failure"]
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

rf_params_bin = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

xgb_params_bin = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.8, 1.0]
}

binary_models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, solver='saga', random_state=42)
}

rf_grid_bin = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                                 rf_params_bin, n_iter=8, cv=3, scoring='f1', random_state=42)
rf_grid_bin.fit(Xb_train, yb_train)
binary_models["RandomForest_Tuned"] = rf_grid_bin.best_estimator_

xgb_grid_bin = RandomizedSearchCV(XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
                                  xgb_params_bin, n_iter=8, cv=3, scoring='f1', random_state=42)
xgb_grid_bin.fit(Xb_train, yb_train)
binary_models["XGBoost_Tuned"] = xgb_grid_bin.best_estimator_

results_binary = {}

for name, model in binary_models.items():
    mlflow.end_run()
    with mlflow.start_run(run_name=f"Binary_{name}"):
        model.fit(Xb_train, yb_train)
        preds = model.predict(Xb_test)
        probs = model.predict_proba(Xb_test)[:, 1] if hasattr(model, "predict_proba") else None
        roc_auc = roc_auc_score(yb_test, probs) if probs is not None else None
        acc = accuracy_score(yb_test, preds)
        prec = precision_score(yb_test, preds, zero_division=0)
        rec = recall_score(yb_test, preds, zero_division=0)
        f1 = f1_score(yb_test, preds, zero_division=0)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        if roc_auc is not None:
            mlflow.log_metric("roc_auc", roc_auc)

        signature = infer_signature(Xb_train, model.predict(Xb_train))
        input_example = Xb_train.iloc[:2]
        mlflow.sklearn.log_model(model, artifact_path=f"Binary_{name}",
                                 signature=signature, input_example=input_example)

        results_binary[name] = {"f1_score": f1, "accuracy": acc}
        print(f"Binary {name}: Accuracy {acc:.3f}, F1 {f1:.3f}, ROC-AUC {roc_auc}")

best_bin = max(results_binary.items(), key=lambda x: x[1]["f1_score"])[0]
print(f"Best Binary Model: {best_bin} (F1-score: {results_binary[best_bin]['f1_score']:.3f})")

# Multiclass classification experiment
mlflow.set_experiment("failure-prediction-multiclass")
X_multi = df.drop(columns=["Type_of_failure", "Machine_failure"], errors="ignore")
y_multi = df["Type_of_failure"]
Xm_train, Xm_test, ym_train, ym_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

rf_params_multi = {
    'n_estimators': [100, 200, 400],
    'max_depth': [6, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

xgb_params_multi = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.8, 1.0]
}

multiclass_models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, solver='saga', n_jobs=-1,
                                             random_state=42, multi_class="multinomial")
}

rf_grid_multi = RandomizedSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                                   rf_params_multi, n_iter=8, cv=3, scoring='f1_weighted', random_state=42)
rf_grid_multi.fit(Xm_train, ym_train)
multiclass_models["RandomForest_Tuned"] = rf_grid_multi.best_estimator_

xgb_grid_multi = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="mlogloss"),
                                    xgb_params_multi, n_iter=8, cv=3, scoring='f1_weighted', random_state=42)
xgb_grid_multi.fit(Xm_train, ym_train)
multiclass_models["XGBoost_Tuned"] = xgb_grid_multi.best_estimator_

results_multi = {}

for name, model in multiclass_models.items():
    mlflow.end_run()
    with mlflow.start_run(run_name=f"Multiclass_{name}"):
        model.fit(Xm_train, ym_train)
        preds = model.predict(Xm_test)

        acc = accuracy_score(ym_test, preds)
        prec = precision_score(ym_test, preds, average="weighted", zero_division=0)
        rec = recall_score(ym_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(ym_test, preds, average="weighted", zero_division=0)

        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})

        signature = infer_signature(Xm_train, model.predict(Xm_train))
        input_example = Xm_train.iloc[:2]
        mlflow.sklearn.log_model(model, artifact_path=f"Multiclass_{name}",
                                 signature=signature, input_example=input_example)

        results_multi[name] = {"f1_score": f1, "accuracy": acc}
        print(f"Multiclass {name}: Accuracy {acc:.3f}, F1 {f1:.3f}")

best_multi = max(results_multi.items(), key=lambda x: x[1]["f1_score"])[0]
print(f"Best Multiclass Model: {best_multi} (F1-score: {results_multi[best_multi]['f1_score']:.3f})")

print("Experiment tracking complete. Use 'mlflow ui' to visualize results.")
