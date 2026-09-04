import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Streamlit page configuration
st.set_page_config(
    page_title="Car Price Prediction & Analytics",
    page_icon="🚗",
    layout="wide"
)

# Hide Streamlit default menu/header/footer
hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown(
    "<h2 style='text-align: center;'>Car Price Prediction</h2>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### Machine Learning Project

This application predicts the selling price of used cars using a
**Random Forest Regression** model.

### Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
""")

# Load the saved model
model_filename = current_dir / "models" / "Random Forest.pkl"

with open(model_filename, "rb") as model_file:
    loaded_model = pickle.load(model_file)

# Load raw data
raw_df = pd.read_csv(
    current_dir / "dataset" / "raw" / "CAR DETAILS.csv"
)

# Load processed data
cleaned_data_filename = (
    current_dir / "dataset" / "processed" / "Processed CAR DETAILS.csv"
)

cleaned_data = pd.read_csv(cleaned_data_filename)

# Categorical columns
category_col = [
    "Brand",
    "Model",
    "Variant",
    "Fuel",
    "Seller_Type",
    "Transmission",
    "Owner"
]

# Main navigation
menu_list = [
    "Raw-Data Display",
    "Exploratory Data Analysis",
    "Predict Selling Price"
]

menu = st.radio("Menu", menu_list)

# ---------------------------------------------------------
# RAW DATA
# ---------------------------------------------------------

if menu == "Raw-Data Display":

    st.title("🚗 Used Car Dataset")

    st.write(
        "Explore the original used-car dataset used for this project."
    )

    if st.checkbox("View Raw Car Dataset"):
        st.dataframe(raw_df, use_container_width=True)


# ---------------------------------------------------------
# EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------

elif menu == "Exploratory Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore the processed dataset used for machine learning."
    )

    if st.checkbox("View Processed Dataset"):
        st.dataframe(cleaned_data, use_container_width=True)

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Cars",
            f"{len(cleaned_data):,}"
        )

    with col2:
        st.metric(
            "Features",
            len(cleaned_data.columns)
        )

    with col3:
        st.metric(
            "Average Selling Price",
            f"₹{cleaned_data['Selling_Price'].mean():,.0f}"
        )


# ---------------------------------------------------------
# PRICE PREDICTION
# ---------------------------------------------------------

elif menu == "Predict Selling Price":

    st.title("🚘 Car Selling Price Prediction")

    st.write(
        "Enter the details of a used car to estimate its selling price."
    )

    # Function for encoding categorical data
    def preprocess_data(df, label_encoders):

        for feature in df.columns:

            if feature in label_encoders:
                df[feature] = label_encoders[feature].transform(
                    df[feature]
                )

        return df

    # Create label encoders
    label_encoders = {}

    for feature in category_col:

        label_encoder = LabelEncoder()

        label_encoder.fit(cleaned_data[feature])

        label_encoders[feature] = label_encoder

    # Display option
    display_option = st.radio(
        "Select Data Display:",
        ["No Data", "Loaded CSV Data", "Encoded Data"]
    )

    # Encode dataset
    encoded_data = preprocess_data(
        cleaned_data.copy(),
        label_encoders
    )

    # Display selected data
    if display_option == "Loaded CSV Data":

        st.subheader("Loaded Dataset")

        st.dataframe(
            cleaned_data,
            use_container_width=True
        )

    elif display_option == "Encoded Data":

        st.subheader("Encoded Dataset")

        st.dataframe(
            encoded_data,
            use_container_width=True
        )

    # User input
    st.subheader("Enter Car Details")

    km_driven = st.slider(
        "Kilometers Driven:",
        min_value=int(cleaned_data["Km_Driven"].min()),
        max_value=int(cleaned_data["Km_Driven"].max()),
        value=int(cleaned_data["Km_Driven"].median())
    )

    year = st.slider(
        "Manufacturing Year:",
        min_value=int(cleaned_data["Year"].min()),
        max_value=int(cleaned_data["Year"].max()),
        value=int(cleaned_data["Year"].median())
    )

    selected_brand = st.selectbox(
        "Brand:",
        cleaned_data["Brand"].unique()
    )

    brand_filtered_df = cleaned_data[
        cleaned_data["Brand"] == selected_brand
    ]

    selected_model = st.selectbox(
        "Model:",
        brand_filtered_df["Model"].unique()
    )

    model_filtered_df = brand_filtered_df[
        brand_filtered_df["Model"] == selected_model
    ]

    selected_variant = st.selectbox(
        "Variant:",
        model_filtered_df["Variant"].unique()
    )

    selected_fuel = st.selectbox(
        "Fuel:",
        cleaned_data["Fuel"].unique()
    )

    selected_seller_type = st.selectbox(
        "Seller Type:",
        cleaned_data["Seller_Type"].unique()
    )

    selected_transmission = st.selectbox(
        "Transmission:",
        cleaned_data["Transmission"].unique()
    )

    selected_owner = st.selectbox(
        "Owner:",
        cleaned_data["Owner"].unique()
    )

    # Create input DataFrame
    input_data = pd.DataFrame({

        "Brand": [selected_brand],

        "Model": [selected_model],

        "Variant": [selected_variant],

        "Year": [year],

        "Km_Driven": [km_driven],

        "Fuel": [selected_fuel],

        "Seller_Type": [selected_seller_type],

        "Transmission": [selected_transmission],

        "Owner": [selected_owner]
    })

    # Encode input
    input_data_encoded = preprocess_data(
        input_data.copy(),
        label_encoders
    )

    # Standardize numerical columns
    scaler = StandardScaler()

    numerical_cols = [
        "Year",
        "Km_Driven"
    ]

    input_data_encoded[numerical_cols] = scaler.fit_transform(
        input_data_encoded[numerical_cols]
    )

    # Prediction
    if st.button(
        "💰 Predict Selling Price",
        type="primary"
    ):

        predicted_price = loaded_model.predict(
            input_data_encoded
        )

        st.success(
            f"Estimated Selling Price: ₹{predicted_price[0]:,.2f}"
        )

        st.info(
            "This prediction is an estimate generated by the "
            "machine learning model and may differ from actual "
            "market prices."
        )