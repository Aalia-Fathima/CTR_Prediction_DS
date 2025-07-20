import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model artifacts
try:
    model_artifacts = joblib.load("ctr_model_complete-2.pkl")
    model = model_artifacts['model']
    label_encoders = model_artifacts['label_encoders']
    frequency_maps = model_artifacts['frequency_maps']
    feature_columns = model_artifacts['feature_columns']
    st.success("Model loaded successfully!")
except:
    st.error("Could not load model artifacts. Please check the file path.")
    st.stop()

st.title("🎯 Ad Click-Through Rate (CTR) Prediction")
st.markdown("Predict whether a user will click on an advertisement")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("User Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        area_income = st.number_input("Area Income ($)", min_value=0.0, max_value=200000.0, value=50000.0)
    
    with col2:
        st.subheader("User Behavior")
        daily_time_spent = st.number_input("Daily Time Spent on Site (minutes)", min_value=0.0, max_value=500.0, value=50.0)
        daily_internet_usage = st.number_input("Daily Internet Usage (minutes)", min_value=0.0, max_value=500.0, value=150.0)
    
    st.subheader("Location & Content")
    col3, col4 = st.columns(2)
    
    with col3:
        # Get available cities and countries from label encoders
        try:
            available_cities = list(label_encoders['city'].classes_)
            available_countries = list(label_encoders['country'].classes_)
            available_ad_topics = list(label_encoders['ad_topic'].classes_)
        except:
            available_cities = ["New York", "Los Angeles", "Chicago"]
            available_countries = ["United States", "Canada", "United Kingdom"]
            available_ad_topics = ["Technology", "Fashion", "Sports"]
        
        city = st.selectbox("City", available_cities)
        country = st.selectbox("Country", available_countries)
    
    with col4:
        ad_topic = st.selectbox("Ad Topic", available_ad_topics)
        
        # Time-based inputs
        hour_of_day = st.slider("Hour of Day", 0, 23, 12)
        day_of_month = st.slider("Day of Month", 1, 31, 15)
        day_of_week = st.slider("Day of Week (0=Monday)", 0, 6, 3)
        month = st.slider("Month", 1, 12, 6)
    
    submitted = st.form_submit_button("Predict CTR")

if submitted:
    try:
        # Prepare input data
        input_data = {}
        
        # Numerical features
        input_data['DailyTime_Spent_on_Site'] = daily_time_spent
        input_data['Age'] = age
        input_data['Area_Income'] = area_income
        input_data['Daily_Internet_Usage'] = daily_internet_usage
        
        # Time features
        input_data['day_of_month'] = day_of_month
        input_data['hour_of_day'] = hour_of_day
        input_data['day_of_week'] = day_of_week
        input_data['month'] = month
        
        # Frequency encoding
        input_data['City_frequency'] = frequency_maps['city'].get(city, 1)  # Default to 1 if not found
        input_data['Country_frequency'] = frequency_maps['country'].get(country, 1)
        
        # Label encoding
        try:
            input_data['City_encoded'] = label_encoders['city'].transform([city])[0]
        except:
            input_data['City_encoded'] = 0  # Default for unknown city
        
        try:
            input_data['Country_encoded'] = label_encoders['country'].transform([country])[0]
        except:
            input_data['Country_encoded'] = 0  # Default for unknown country
        
        try:
            input_data['Ad_Topic_encoded'] = label_encoders['ad_topic'].transform([ad_topic])[0]
        except:
            input_data['Ad_Topic_encoded'] = 0  # Default for unknown ad topic
        
        # Gender encoding
        input_data['Gender_encoded'] = 1 if gender == "Male" else 0
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all features are present and in correct order
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Display results
        st.success("Prediction completed!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.success("🎯 **WILL CLICK**")
            else:
                st.warning("❌ **WON'T CLICK**")
        
        with col2:
            st.metric("Click Probability", f"{probability:.1%}")
        
        # Additional insights
        st.subheader("📊 Insights")
        if probability > 0.7:
            st.success("🎯 **Excellent targeting!** This user profile shows high engagement potential.")
        elif probability > 0.5:
            st.warning("⚠️ **Moderate targeting.** Consider optimizing ad content or timing.")
        else:
            st.error("❌ **Poor targeting.** This user profile is unlikely to engage.")
        
        # Show feature importance
        with st.expander("Feature Importance"):
            if 'feature_importance' in model_artifacts:
                st.dataframe(model_artifacts['feature_importance'].head(10))
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check your input values and try again.")

