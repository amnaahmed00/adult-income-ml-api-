# Adult Income Prediction using Machine Learning

A machine learning project that predicts income categories (`<=50K` or `>50K`) using the UCI Adult dataset.

## Models and Results

| Model | Accuracy |
|---|---:|
| LightGBM | 87.37% |
| Random Forest | 85.09% |
| Logistic Regression | 85.07% |
| Dummy Classifier | 76.38% |

## Features

- Data cleaning and preprocessing
- Model training and evaluation
- Confusion matrix visualizations
- FastAPI prediction endpoints for all four models

## Project Structure
<pre>
app.py
src/adult_income/
  ├── data.py
  ├── preprocessing.py
  ├── train.py
  └── tune.py
models/
frontend/</pre>

## Run the API

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to test the API.

## Dataset

[UCI Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)

**Technologies:** Python, Pandas, Scikit-learn, LightGBM, FastAPI
