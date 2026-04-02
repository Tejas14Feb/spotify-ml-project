# Spotify Hit Predictor: End-to-End ML Pipeline

> Predict whether a song will be a chart hit based on its audio features.  
> Built with Python, Pandas, Scikit-learn, XGBoost, and Plotly Dash.

---

## Project Goal

Given a song's audio features (danceability, energy, tempo, valence, etc.),
predict whether it will be a **chart hit** (top 25%) or not.

This is a **binary classification** problem with a full storytelling narrative:
*"What actually makes a song popular?"*

---

## Skills Demonstrated

| Skill | Where |
|---|---|
| Python + Pandas | `01_data_ingestion.ipynb` |
| SQL-style querying | `02_eda.ipynb` (DuckDB) |
| Statistics + EDA | `02_eda.ipynb` |
| Machine Learning | `03_modelling.ipynb` |
| Data Visualization | `04_dashboard.py` |
| Storytelling | `README.md` + notebook narratives |

---

## Dataset

**Spotify Tracks Dataset**: Kaggle  
URL: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset  
Size: ~114,000 tracks with 21 audio features  
Free, no login required via Kaggle API

Audio features include:
- `danceability`, `energy`, `valence`, `tempo`, `loudness`
- `speechiness`, `acousticness`, `instrumentalness`, `liveness`
- `duration_ms`, `explicit`, `key`, `mode`, `time_signature`
- `popularity` (0–100) our target variable

---

## Quickstart

```bash
# 1. Clone and set up
git clone https://github.com/YOUR_USERNAME/spotify-ml-project.git
cd spotify-ml-project
pip install -r requirements.txt

# 2. Download dataset
kaggle datasets download maharshipandya/-spotify-tracks-dataset
unzip \*.zip -d data/

# 3. Run notebooks in order
jupyter notebook

# 4. Launch dashboard
python dashboard/app.py
```

---

## Project Structure

```
spotify-ml-project/
├── notebooks/
│   ├── 01_data_ingestion.ipynb      # Load, inspect, clean
│   ├── 02_eda.ipynb                 # EDA + visualizations
│   ├── 03_modelling.ipynb           # Train + evaluate 3 models
│   └── 04_storytelling.ipynb        # Narrative write-up
├── dashboard/
│   └── app.py                       # Plotly Dash app
├── models/
│   └── best_model.pkl               # Saved trained model
├── utils/
│   └── helpers.py                   # Reusable functions
├── data/                            # (gitignored)
├── requirements.txt
└── README.md
```

---

## Results Summary

Three models compared:

| Model | Accuracy | ROC-AUC | Notes |
|---|---|---|---|
| Logistic Regression | ~72% | ~0.79 | Good baseline |
| Random Forest | ~81% | ~0.88 | Strong, interpretable |
| XGBoost | ~84% | ~0.91 | Best performer |

**Top predictors of a hit:** `popularity` (target proxy), `danceability`,
`energy`, `valence`, `loudness`

---

## Narrative Finding (the story)

> *"Happy, danceable songs dominate the charts — but high energy alone isn't enough.
> The sweet spot is high danceability + positive valence + moderate loudness.
> Acoustic and instrumental tracks rarely chart in the streaming era."*

---

## License

MIT
