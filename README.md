# Fraud Detection Automation

Automated ML pipeline that trains fraud detection models daily at 2 AM using GitHub Actions.

## Features

- **Daily Training** - Automatically runs at 2 AM UTC
- **5 ML Models** - Logistic Regression, Naive Bayes, Decision Tree, Random Forest, XGBoost
- **MongoDB Integration** - Pulls latest data and stores trained models
- **Metrics & Visualizations** - ROC curves, confusion matrices, feature importance

## Setup

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/automated-fraud-detection-pipeline.git
cd automated-fraud-detection-pipeline
```

### 2. Local Development (Optional)

Create `.env` file:
```bash
MONGO_URI=your_mongodb_connection_string
```

Install dependencies and run:
```bash
pip install -r requirements.txt
python train_model.py
```

### 3. GitHub Actions Setup

Add MongoDB secret:
1. Go to **Settings** � **Secrets and variables** � **Actions**
2. Click **New repository secret**
3. Name: `MONGO_URI`
4. Value: Your MongoDB connection string
5. Save

### 4. Run Training

**Automatic:** Runs daily at 2 AM UTC

**Manual:** Go to **Actions** � **Daily ML Model Training** � **Run workflow**

## How It Works

1. Connects to MongoDB
2. Pulls latest dataset (status: "ready", isActive: true)
3. Trains 5 models
4. Uploads models to MongoDB GridFS
5. Saves metrics and visualizations

## Files

- `train_model.py` - Main training script
- `.github/workflows/daily_training.yml` - GitHub Actions workflow
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (local only, not committed)

## Troubleshooting

**Workflow fails?**
- Check `MONGO_URI` secret is set
- Verify dataset exists in MongoDB with status "ready"
- Check workflow logs in Actions tab

**Local run fails?**
- Make sure `.env` file exists with correct MongoDB URI
- Install all dependencies: `pip install -r requirements.txt`

