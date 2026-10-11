# Car Price Prediction

A machine learning project that estimates the selling price of used cars based on the details provided by the user. The project includes dataset exploration, analysis, and a Streamlit interface for generating predictions.

## Project Overview

The objective of this project is to explore used-car data and use a trained machine learning model to estimate car selling prices. Users can enter car details into the web application and view the predicted price.

## Features

- **Dataset Exploration:** View the available car dataset.
- **Exploratory Data Analysis:** Explore the dataset and identify patterns in the data.
- **Data Processing:** Use processed data for model-based predictions.
- **Price Prediction:** Generate estimated selling prices using a saved Random Forest model.
- **Interactive Web App:** Access the prediction functionality through Streamlit.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Regression
- Streamlit

## Project Structure

```text
car-price-prediction/
├── dataset/
│   └── processed/
│       └── Processed CAR DETAILS.csv
├── models/
│   └── Random Forest.pkl
├── notebooks/
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/mayankpal19/car-price-prediction.git
cd car-price-prediction
```

### 2. Install the required packages

Make sure Python is installed, then run:

```bash
python -m pip install -r requirements.txt
```

### 3. Start the Streamlit application

```bash
python -m streamlit run streamlit_app.py
```

Open the following address in your browser:

`http://localhost:8501`

## How the Application Works

1. The user opens the Streamlit application.
2. The user enters the required car details.
3. The application processes the input into the format expected by the saved model.
4. The Random Forest model generates a predicted selling price.
5. The application displays the prediction.

## Model Information

The application uses a saved Random Forest model for price prediction. The model file is stored in the `models` directory.

## Limitations

The predicted selling price is an estimate based on the model and its training data. Actual market prices may differ depending on vehicle condition, location, demand, and other factors.

## Acknowledgements

This project uses machine learning techniques for educational purposes. If code, notebooks, or model components were adapted from another project or tutorial, the original source should be credited accordingly.
