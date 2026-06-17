# FlowSense AI - Traffic Demand Predictor

A beautiful, production-ready full-stack web application that predicts traffic demand based on geospatial, temporal, and environmental features using an ensemble ML model.


## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Tech Stack](#-tech-stack)
4. [Project Structure](#-project-structure)
5. [Dataset Information](#-dataset-information)
6. [Machine Learning Model](#-machine-learning-model)
7. [Installation & Local Setup](#-installation--local-setup)
8. [Deploying to Production](#-deploying-to-production)
9. [Usage Guide](#-usage-guide)
10. [Contributing](#-contributing)

---

## 🚀 Project Overview

FlowSense AI is a sophisticated traffic demand forecasting system designed to help urban planners, traffic management centers, and mobility service providers predict traffic volume at specific locations and times. It combines multiple machine learning models to achieve an ensemble approach for superior predictive accuracy, and features a modern, intuitive user interface.

---

## ✨ Features

- 🎯 **Core Prediction Engine**
  - Ensemble of LightGBM, XGBoost, and CatBoost regressors
  - Cross-validated R² score of ~0.96
  - Target encoding for categorical features
  - Handles missing data gracefully

- 🗺️ **Geospatial Intelligence**
  - Decodes geohashes to latitude & longitude for spatial analysis
  - Uses geohash-based target encoding

- 🌦️ **Environmental Context**
  - Weather-aware predictions
  - Temperature-based adjustments

- 📱 **Beautiful UI/UX**
  - Responsive design, mobile-first interface
  - Real-time prediction with smooth animations
  - Local storage of prediction history
  - Beautiful gradient UI with glassmorphism styling

- 🔧 **Production-Ready**
  - Easy deployable with Render (free & paid tiers available)
  - Node.js + Express backend
  - React + Vite frontend
  - Single deployment from a single repository


---

## 🛠️ Tech Stack

### Frontend
| Technology | Purpose |
|--------------|---------|
| React 18 | UI framework |
| Vite 5 | Build tool & dev server |
| TypeScript | Type safety |
| Tailwind CSS 3 | Utility-first styling |
| PostCSS | CSS transformations |

### Backend
| Technology | Purpose |
|------------|---------|
| Node.js 18+ | Runtime environment |
| Express 4 | RESTful API server |
| CORS | Cross-origin requests |

### Machine Learning
| Technology | Purpose |
|------------|---------|
| Python 3.9+ | ML runtime |
| LightGBM 4.6 | Gradient boosting model |
| XGBoost 2.1 | Gradient boosting model |
| CatBoost 1.2 | Gradient boosting model |
| Pandas | Data manipulation & processing |
| Scikit-learn 1.6 | ML utilities |
| Joblib | Model serialization |
| Geohash2 | Geospatial encoding/decoding |

---

## 📁 Project Structure

```
AuroraPulse/
├── client/                 # React Frontend
│   ├── src/
│   │   ├── App.tsx    # Main UI Component
│   │   ├── main.tsx    # Vite entry point
│   │   └── styles.css  # Global styles
│   ├── index.html       # Vite HTML entry
│   ├── vite.config.ts  # Vite build config
│   ├── tsconfig.json   # TypeScript config
│   ├── tailwind.config.js
│   └── package.json
│
├── server/                # Express Backend
│   └── index.js        # Express server (serves frontend + API)
│   └── package.json
│
├── ml/                   # Machine Learning
│   ├── train.py        # Model training script
│   ├── predict.py      # Inference script
│   ├── requirements.txt
│   └── artifacts/      # Trained model & metadata (generated)
│
├── train.csv            # Training dataset
├── test.csv             # Test dataset
├── sample_submission.csv
├── render.yaml          # Render deployment config
├── build.js            # Build script
└── package.json        # Root package.json
```

---

## 📊 Dataset Information

### Dataset Overview
The dataset contains traffic demand observations across multiple locations, time periods, and environmental conditions.

### Training Dataset
- **Total Rows**: 77,299 rows
- **Features**: 10 input features + 1 target column
- **Target**: `demand` (normalized traffic volume)

### Test Dataset
- **Total Rows**: Varies
- **Features**: 10 input features
- **Target**: To predict

### Features Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `geohash` | String | Encoded geographic location |
| `day` | Integer | Day number (time series) |
| `timestamp` | String | Time in "H:MM" format |
| `RoadType` | Categorical | Type of road (Residential, Street, Highway, etc.) |
| `NumberofLanes` | Integer | Number of lanes at location |
| `LargeVehicles` | Categorical | Whether large vehicles (trucks, etc.) are allowed (Allowed/Not Allowed) |
| `Landmarks` | Categorical | Whether there is a landmark nearby (Yes/No) |
| `Temperature` | Float | Ambient temperature at location |
| `Weather` | Categorical | Weather condition (Sunny, Rainy, Foggy, Snowy) |

---

## 🤖 Machine Learning Model

### Model Architecture
We use an **ensemble of gradient-boosted trees**:
- LightGBM (40% weight)
- XGBoost (30% weight)
- CatBoost (30% weight)

### Feature Engineering
1. **Temporal Features**
   - `hour`: Hour of day from timestamp
   - `minute`: Minute of hour
   - `total_minutes`: Total minutes since midnight

2. **Spatial Features**
   - Decodes geohash → lat/lng
   - Target encoding using geohash
   - Target encoding using timestamp
   - Target encoding using geohash-hour combinations

3. **Categorical Encoding**
   - RoadType, Weather, LargeVehicles, Landmarks: Label encoding

### Model Performance
- **Cross-Validated R² Score**: **0.9592** (≈95.92% variance explained)

---

## 🛠️ Installation & Local Setup

### Prerequisites
- Node.js 18+
- Python 3.9+
- pip
- npm/yarn/pnpm

### Step 1: Clone Repo
```bash
git clone https://github.com/[your-username]/FlowSense-AI.git
cd FlowSense-AI
```

### Step 2: Python Setup (Virtual Environment)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate    # Windows
pip install -r ml/requirements.txt
```

### Step 3: Train the ML Model
```bash
python3 ml/train.py
```

### Step 4: Install Node Dependencies
```bash
cd client && npm install
cd ../server && npm install
cd ..
```

### Step 5: Build Frontend
```bash
cd client && npm run build
```

### Step 6: Start the Application
```bash
cd server && npm start
```

The application will now be running on **http://localhost:5175**

---

## 🌐 Deploying to Production

### Deploying on Render (Recommended)

Render offers an easy way to deploy everything from one repo for free!

1. **Sign up for a free [Render](https://render.com) account**
2. Connect your GitHub repository
3. Click "New Web Service" from the dashboard
4. Use the pre-configured `render.yaml`
5. Let Render do the work:
   - Build and deploy everything automatically
   - Train the ML model during build
   - Serve your app on an HTTPS URL

### Render Configuration (`render.yaml`)
```yaml
services:
  - type: web
    name: traffic-demand-predictor
    env: node
    plan: free
    buildCommand: node build.js
    startCommand: cd server && npm start
    envVars:
      - key: NODE_ENV
        value: production
```

---

## 📱 Usage Guide

### Step-by-Step Prediction
1. Enter or select a **geohash** from the suggestions
2. Select **day**, **timestamp**, and **road type**
3. Configure number of lanes, large vehicle allowance, landmark presence
4. Enter the weather & temperature
5. Click **Generate Estimate**

### Local History
All predictions are automatically saved to your browser's localStorage for easy access later!

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an issue or pull request.

---

## 📄 License
MIT License - use this project for personal or commercial use.
