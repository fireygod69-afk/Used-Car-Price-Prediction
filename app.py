import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Har model ko import karein
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# --- MODEL TRAINING & SCORING (Cached) ---
@st.cache_resource
def load_and_train_models():
    # 1. Data Load & Clean
    df = pd.read_csv('car details v4.csv.xls')
    columns_to_keep = ['Year', 'Kilometer', 'Fuel Type', 'Transmission', 'Engine', 'Max Power', 'Price']
    df = df[columns_to_keep]

    df['Engine'] = pd.to_numeric(df['Engine'].astype(str).str.split(' ').str[0], errors='coerce')
    df['Max Power'] = pd.to_numeric(df['Max Power'].astype(str).str.split(' ').str[0], errors='coerce')
    df = df.dropna()

    df = pd.get_dummies(df, columns=['Fuel Type', 'Transmission'], drop_first=True, dtype=int)

    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # 2. Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Har Model Aur Score Save Karein
    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
        # R2 score ko percentage mein convert karke save kar rahe hain
        model_scores[name] = r2_score(y_test, predictions) * 100
        
    # Prediction ke liye sabse best model (Random Forest) ko return karein
    best_model = models["Random Forest"]
    
    return best_model, scaler, X.columns, model_scores

# Backend se data load kar rahe hain
best_model, scaler, feature_columns, scores = load_and_train_models()


# --- WEBSITE FRONTEND ---
st.title("🚗 Used Car Price Predictor")
st.write("Apni car ki details niche dalein aur uska estimated price janiye!")

# --- NAYA SECTION: MODEL ACCURACY DASHBOARD ---
# Ek expandable box banaya hai jisme har model ka score beautifully dikhega
with st.expander("📊 View Model Accuracies (R-Squared Scores)", expanded=False):
    st.write("Hamne is data par alag-alag regression models test kiye hain. Yahan unki performance hai:")
    
    # Scores ko ek line mein dikhane ke liye 4 columns banaye
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Linear Reg.", f"{scores['Linear Regression']:.2f}%")
    col2.metric("KNN", f"{scores['KNN Regressor']:.2f}%")
    col3.metric("Decision Tree", f"{scores['Decision Tree']:.2f}%")
    col4.metric("Random Forest", f"{scores['Random Forest']:.2f}%", delta="Best Model")
    
st.markdown("---")

# --- USER INPUTS ---
col_a, col_b = st.columns(2)

with col_a:
    car_age = st.number_input("Car Age (Years)", min_value=0, max_value=30, value=5)
    car_km = st.number_input("Kilometers Driven", min_value=0, value=45000, step=1000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])

with col_b:
    car_engine = st.number_input("Engine (CC)", min_value=500, value=1197, step=100)
    car_power = st.number_input("Max Power (bhp)", min_value=10, value=82, step=5)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.markdown("---")

# --- PREDICTION BUTTON ---
if st.button("Predict Price", type="primary"):
    
    car_year = 2026 - car_age
    input_data = {col: [0] for col in feature_columns}
    user_df = pd.DataFrame(input_data)
    
    user_df['Year'] = car_year
    user_df['Kilometer'] = car_km
    user_df['Engine'] = car_engine
    user_df['Max Power'] = car_power
    
    if fuel_type == "Petrol" and 'Fuel Type_Petrol' in user_df.columns:
        user_df['Fuel Type_Petrol'] = 1
    elif fuel_type == "Diesel" and 'Fuel Type_Diesel' in user_df.columns:
        user_df['Fuel Type_Diesel'] = 1
        
    if transmission == "Manual" and 'Transmission_Manual' in user_df.columns:
        user_df['Transmission_Manual'] = 1

    try:
        user_df_scaled = scaler.transform(user_df)  
        predicted_price = best_model.predict(user_df_scaled)
        
        st.success("Prediction Successful! 🎉")
        st.metric(label="Estimated Selling Price (via Random Forest)", value=f"₹ {predicted_price[0]:,.2f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")