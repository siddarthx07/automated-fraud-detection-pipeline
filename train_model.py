import os
import time
import joblib
import gridfs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for GitHub Actions
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from datetime import datetime
from pymongo import MongoClient
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load environment variables from .env file (for local development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, assume env vars are set (e.g., in GitHub Actions)
    pass

# ======================================
# MongoDB Connection
# ======================================
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("❌ MONGO_URI environment variable not set! Please check your .env file or GitHub secrets.")
client = MongoClient(mongo_uri)
db = client["mccs"]
fs = gridfs.GridFS(db, collection="modelFiles")
mlmodels_collection = db["mlmodels"]

# ======================================
# Classifiers
# ======================================
classifiers = {
    "logistic regression": LogisticRegression(),
    "bernoulli nb": BernoulliNB(),
    "decision tree": DecisionTreeClassifier(),
    "random forest": RandomForestClassifier(),
    "xgboost pipeline": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
}

# ======================================
# Data Cleaning Function
# ======================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
    df["Transaction Day"] = df["Transaction Date"].dt.day
    df["Transaction DOW"] = df["Transaction Date"].dt.day_of_week
    df["Transaction Month"] = df["Transaction Date"].dt.month

    mean_value = np.round(df["Customer Age"].mean(), 0)
    df["Customer Age"] = np.where(df["Customer Age"] <= -9, np.abs(df["Customer Age"]), df["Customer Age"])
    df["Customer Age"] = np.where(df["Customer Age"] < 9, mean_value, df["Customer Age"])

    df["Is Address Match"] = (df["Shipping Address"] == df["Billing Address"]).astype(int)
    df.drop(columns=["Transaction ID", "Customer ID", "Customer Location",
                     "IP Address", "Transaction Date", "Shipping Address", "Billing Address"],
            inplace=True, errors="ignore")

    int_col = df.select_dtypes(include="int").columns
    float_col = df.select_dtypes(include="float").columns
    df[int_col] = df[int_col].apply(pd.to_numeric, downcast="integer")
    df[float_col] = df[float_col].apply(pd.to_numeric, downcast="float")
    return df

# ======================================
# Helper Functions
# ======================================
def get_latest_dataset_version():
    datasets = list(db.trainingdatasets.find({
        "isActive": True,
        "status": "ready"
    }))
    if not datasets:
        raise Exception("❌ No active dataset found in MongoDB.")

    def parse_version(version_str):
        return tuple(int(x) for x in version_str.replace('v', '').split('.'))

    latest = max(datasets, key=lambda d: parse_version(d["version"]))
    print(f"📊 Using dataset version: {latest['version']}")
    return latest

def pull_data(dataset_id, dataset_version, data_type):
    print(f"📥 Pulling {data_type.upper()} data...")
    cursor = db.trainingdatas.find({
        "datasetId": dataset_id,
        "datasetVersion": dataset_version,
        "dataType": data_type
    })
    df = pd.DataFrame(list(cursor))
    if df.empty:
        raise Exception(f"❌ No {data_type} data found in MongoDB for version {dataset_version}")

    # Drop MongoDB metadata fields
    df = df.drop(columns=['_id', 'datasetId', 'datasetVersion', 'dataType'], errors='ignore')

    print(f"✅ Loaded {len(df)} records ({data_type})")
    return df

# ======================================
# Store model metrics and visualizations
# ======================================
def store_model_data(model_name, classifier, X_test, y_test, train_data, version):
    metrics = []
    plots = []
    additional_info = {}

    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    metrics.append({"name": "Accuracy", "value": accuracy})

    cm = confusion_matrix(y_test, y_pred)
    metrics.append({"name": "Confusion Matrix", "value": cm.tolist()})

    cr = classification_report(y_test, y_pred, output_dict=True)
    metrics.append({"name": "Classification Report", "value": cr})

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
    fig_roc.update_layout(title=f'ROC Curve (AUC = {roc_auc:.2f})',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate')
    plots.append({"type": "ROC Curve", "data": fig_roc.to_json()})

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    fig_pr = px.area(x=recall, y=precision,
                     title=f'Precision-Recall Curve (AP = {avg_precision:.2f})',
                     labels={"x": "Recall", "y": "Precision"})
    plots.append({"type": "Precision-Recall Curve", "data": fig_pr.to_json()})

    # Confusion Matrix Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plots.append({"type": "Confusion Matrix Heatmap", "data": image_base64})
    plt.close()

    # Feature Importance (Decision Tree, RF, XGB)
    try:
        model = classifier.named_steps['classifier']
        if hasattr(model, "feature_importances_"):
            feature_importances = model.feature_importances_
            feature_names = classifier.named_steps['transformer'].get_feature_names_out(input_features=train_data.columns)
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False).head(20)  # Top 20 features
            additional_info["feature_importance"] = importance_df.to_dict(orient='records')

            fig_fi = px.bar(importance_df, x="Importance", y="Feature", orientation='h',
                           title="Top 20 Feature Importances")
            plots.append({"type": "Feature Importance", "data": fig_fi.to_json()})
    except Exception as e:
        print(f"⚠️ Feature importance not available for {model_name}: {e}")

    # Pretty naming (for metrics)
    pretty_names = {
        "logistic regression": "Logistic Regression",
        "bernoulli nb": "Bernoulli NB",
        "decision tree": "Decision Tree",
        "random forest": "Random Forest",
        "xgboost pipeline": "XGBoost"
    }

    model_doc = {
        "modelName": pretty_names.get(model_name.lower(), model_name.title()),
        "modelType": "Classification",
        "version": version,
        "metrics": metrics,
        "plots": plots,
        "createdAt": datetime.utcnow(),
        "additionalInfo": additional_info
    }

    mlmodels_collection.insert_one(model_doc)
    print(f"📦 Stored {model_doc['modelName']} (v{version}) metrics & visualizations!")

# ======================================
# Main Training Logic
# ======================================
def main():
    print("=" * 80)
    print(f"🤖 AUTOMATED TRAINING STARTED: {datetime.now()}")
    print("=" * 80)

    latest_dataset = get_latest_dataset_version()
    dataset_id = latest_dataset["_id"]
    dataset_version = latest_dataset["version"]

    train_df = pull_data(dataset_id, dataset_version, "train")
    test_df = pull_data(dataset_id, dataset_version, "test")

    print("\n🧹 Cleaning & preprocessing data...")
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    train_data = train_df.drop(columns=["Is Fraudulent"])
    train_label = train_df["Is Fraudulent"]
    test_data = test_df.drop(columns=["Is Fraudulent"])
    test_label = test_df["Is Fraudulent"]

    cat_col = train_data.select_dtypes(include="O").columns
    num_col = [col for col in train_data.columns if col not in cat_col and col != "Is Address Match"]

    transformer = ColumnTransformer(transformers=[
        ('encoding', OneHotEncoder(handle_unknown='ignore'), cat_col),
        ('scaling', StandardScaler(), num_col)
    ], remainder='passthrough')

    os.makedirs("models", exist_ok=True)
    results = []

    for name, clf in classifiers.items():
        print(f"\n🚀 Training {name}...")
        start = time.time()

        model = Pipeline(steps=[
            ("transformer", transformer),
            ("classifier", clf)
        ])
        model.fit(train_data, train_label)

        acc = accuracy_score(test_label, model.predict(test_data))
        duration = time.time() - start
        results.append((name, acc, duration))

        # Save model locally
        filename = f"{name.replace(' ', '_')}.pkl"
        path = os.path.join("models", filename)
        joblib.dump(model, path)

        # Determine version
        new_version = "v1.0.0"
        existing = list(db.modelFiles.files.find({"base_name": filename}).sort("version", -1))
        if existing and "version" in existing[0]:
            last_ver = existing[0]["version"]
            parts = last_ver.replace("v", "").split(".")
            new_version = f"v{parts[0]}.{int(parts[1]) + 1}.0"

        # Upload to GridFS
        with open(path, "rb") as f:
            file_id = fs.put(
                f,
                filename=filename,
                base_name=filename,
                version=new_version,
                trained_on_dataset_version=dataset_version,
                upload_time=datetime.utcnow(),
                accuracy=float(acc)
            )

        # Store metrics + plots
        store_model_data(name, model, test_data, test_label, train_data, new_version)

        print(f"   ✅ Accuracy: {acc:.4f} | Uploaded {filename} (v{new_version})")

    print("\n" + "=" * 80)
    print("🎯 TRAINING COMPLETED SUCCESSFULLY!\n")
    for n, a, t in results:
        print(f"   {n:25s} | Accuracy: {a:.4f} | Time: {t:.2f}s")
    print("=" * 80)

    client.close()

if __name__ == "__main__":
    main()
