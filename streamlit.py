import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Set page config
st.set_page_config(page_title="Sleep Hours Predictor", page_icon="😴", layout="wide")

# Title and description
st.title("😴 Sleep Hours Predictor")
st.markdown("---")
st.write("Enter your daily parameters below to predict your optimal sleep hours!")

# Load and train model (you can cache this for better performance)
@st.cache_data
def load_and_train_model():
    # Load the dataset
    DATA_PATH = 'data.csv'
    data = pd.read_csv(DATA_PATH)
    
    # Features and target
    X = data.drop('sleep_hours', axis=1)
    y = data['sleep_hours']
    
    # Preprocess categorical features
    categorical = ['type_of_work', 'medical_conditions']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ], remainder='passthrough')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform features
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_trans, y_train)
    
    # Calculate model performance
    y_pred = model.predict(X_test_trans)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, preprocessor, mae, rmse

# Load model
try:
    model, preprocessor, mae, rmse = load_and_train_model()
    model_loaded = True
except FileNotFoundError:
    st.error("❌ Error: 'data.csv' file not found. Please make sure the data file is in the same directory.")
    model_loaded = False
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    model_loaded = False

if model_loaded:
    # Display model performance
    with st.expander("📊 Model Performance"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f} hours")
        with col2:
            st.metric("Root Mean Square Error", f"{rmse:.2f} hours")

    # Create input form
    st.subheader("📝 Enter Your Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*Physical & Health Data*")
        physical_activity_hours = st.number_input(
            "Physical Activity Hours", 
            min_value=0.0, 
            max_value=24.0, 
            value=1.0, 
            step=0.5,
            help="Hours spent on physical activities per day"
        )
        
        age = st.number_input(
            "Age", 
            min_value=1, 
            max_value=120, 
            value=25,
            help="Your age in years"
        )
        
        weight_kg = st.number_input(
            "Weight (kg)", 
            min_value=30.0, 
            max_value=300.0, 
            value=70.0, 
            step=0.5,
            help="Your weight in kilograms"
        )
        
        medical_conditions = st.selectbox(
            "Medical Conditions",
            options=["none", "hypertension", "asthma"],
            help="Select any existing medical conditions"
        )
        
        hydration_liters = st.number_input(
            "Daily Hydration (Liters)", 
            min_value=0.0, 
            max_value=10.0, 
            value=2.0, 
            step=0.1,
            help="Amount of water consumed per day"
        )
        
        number_of_meals = st.number_input(
            "Number of Meals", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Number of meals consumed per day"
        )
    
    with col2:
        st.markdown("*Lifestyle & Work Data*")
        type_of_work = st.selectbox(
            "Type of Work",
            options=["mental", "physical", "remote", "mixed"],
            help="Primary type of work you do"
        )
        
        screen_time_hours = st.number_input(
            "Screen Time Hours", 
            min_value=0.0, 
            max_value=24.0, 
            value=6.0, 
            step=0.5,
            help="Hours spent looking at screens per day"
        )
        
        commute_time_minutes = st.number_input(
            "Commute Time (Minutes)", 
            min_value=0, 
            max_value=300, 
            value=30,
            help="Daily commute time in minutes"
        )
        
        stress_level = st.slider(
            "Stress Level", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Rate your stress level from 1 (low) to 10 (high)"
        )
        
        mood_rating = st.slider(
            "Mood Rating", 
            min_value=1, 
            max_value=10, 
            value=7,
            help="Rate your overall mood from 1 (poor) to 10 (excellent)"
        )
        
        alcohol_caffeine_units = st.number_input(
            "Alcohol/Caffeine Units", 
            min_value=0, 
            max_value=20, 
            value=2,
            help="Daily units of alcohol or caffeine consumed"
        )

    # Prediction button
    st.markdown("---")
    if st.button("🔮 Predict Sleep Hours", type="primary", use_container_width=True):
        # Create user data dictionary
        user_data = {
            'physical_activity_hours': physical_activity_hours,
            'type_of_work': type_of_work,
            'screen_time_hours': screen_time_hours,
            'stress_level': stress_level,
            'number_of_meals': number_of_meals,
            'hydration_liters': hydration_liters,
            'commute_time_minutes': commute_time_minutes,
            'age': age,
            'weight_kg': weight_kg,
            'medical_conditions': medical_conditions,
            'alcohol_caffeine_units': alcohol_caffeine_units,
            'mood_rating': mood_rating
        }
        
        # Convert to DataFrame and make prediction
        user_df = pd.DataFrame([user_data])
        user_trans = preprocessor.transform(user_df)
        user_pred = model.predict(user_trans)
        
        # Display prediction
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        predicted_hours = user_pred[0]
        hours = int(predicted_hours)
        minutes = int((predicted_hours - hours) * 60)
        
        # Create attractive result display
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Recommended Sleep Duration",
                value=f"{hours} hours {minutes} minutes",
                delta=f"{predicted_hours:.2f} hours total"
            )
        
        # Provide sleep recommendations based on prediction
        st.subheader("💡 Sleep Recommendations")
        
        if predicted_hours < 6:
            st.warning("⚠ Your predicted sleep duration is quite low. Consider:")
            recommendations = [
                "Reducing screen time before bed",
                "Managing stress through relaxation techniques",
                "Creating a consistent bedtime routine",
                "Consulting with a healthcare professional"
            ]
        elif predicted_hours > 9:
            st.info("ℹ Your predicted sleep duration is on the higher side. This might indicate:")
            recommendations = [
                "Good sleep hygiene habits",
                "Physical activity promoting deeper rest",
                "Consider if you're getting quality sleep",
                "Monitor for any underlying sleep disorders"
            ]
        else:
            st.success("✅ Your predicted sleep duration looks healthy! Keep up the good habits:")
            recommendations = [
                "Maintain your current routine",
                "Continue regular physical activity",
                "Keep stress levels manageable",
                "Stay consistent with your sleep schedule"
            ]
        
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            st.write("*Your Input Data:*")
            for key, value in user_data.items():
                formatted_key = key.replace('_', ' ').title()
                st.write(f"• *{formatted_key}:* {value}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Sleep Hours Predictor • Built with Streamlit 💤"
    "</div>", 
    unsafe_allow_html=True
)