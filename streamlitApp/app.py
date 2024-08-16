import numpy as np
import streamlit as st
import pickle

st.title('Customer Churn Automation')

try:
    model = pickle.load(open('best_model.pkl', 'rb'))
    scaler = pickle.load(open('scalar.pkl', 'rb'))
    label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()


def encode_features(features, encoders):
    encoded_features = []
    for feature_name, value in features.items():
        if feature_name in encoders:
            encoder = encoders[feature_name]
            if value in encoder.classes_:
                encoded_value = encoder.transform([value])[0]
                print(f"Encoding '{feature_name}': '{value}' -> {encoded_value}")
            else:
                encoded_value = value  # or use a placeholder if necessary
                print(f"Unseen category for '{feature_name}': '{value}' -> {encoded_value}")
            encoded_features.append(int(encoded_value))  # Convert to plain integer
        else:
            encoded_features.append(value)
            print(f"No encoder for '{feature_name}': Using original value {value}")
    return encoded_features


def predict(features):
    print("Raw Features: ", features)

    features_encoded = encode_features(features, label_encoders)
    print("Encoded features: ", features_encoded)

    features_encoded = np.array(features_encoded).reshape(1, -1)  # Ensure 2D shape
    print("Features reshaped for scaling: ", features_encoded)

    features_scaled = scaler.transform(features_encoded)
    print("Scaled Features: ", features_scaled)

    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)

    print("Prediction: ", prediction)
    print("Prediction probabilities: ", prediction_proba)

    return prediction[0], prediction_proba[0]


st.write("Enter the Customer Details")

if 'features' not in st.session_state:
    st.session_state.features = {}

col1, col2, col3 = st.columns(3)

with col1:
    st.session_state.features['Senior Citizen'] = st.selectbox('Senior Citizen', options=['No', 'Yes'])
    st.session_state.features['Partner'] = st.selectbox('Partner', options=['Yes', 'No'])
    st.session_state.features['Dependents'] = st.selectbox('Dependents', options=['No', 'Yes'])
    st.session_state.features['Tenure Months'] = st.number_input('Total Tenure Months', min_value=0, placeholder='Minimum is 0')
    st.session_state.features['Multiple Lines'] = st.selectbox('Multiple Lines', options=['No', 'Yes', 'No phone service'])
    st.session_state.features['Internet Service'] = st.selectbox('Internet Service', options=['Fiber optic', 'No', 'DSL'])

with col2:
    st.session_state.features['Online Security'] = st.selectbox('Online Security', options=['No', 'No internet service', 'Yes'])
    st.session_state.features['Online Backup'] = st.selectbox('Online Backup', options=['No', 'No internet service', 'Yes'])
    st.session_state.features['Device Protection'] = st.selectbox('Device Protection', options=['Yes', 'No internet service', 'No'])
    st.session_state.features['Tech Support'] = st.selectbox('Tech Support', options=['Yes', 'No internet service', 'No'])
    st.session_state.features['Streaming TV'] = st.selectbox('Streaming TV', options=['Yes', 'No internet service', 'No'])
    st.session_state.features['Streaming Movies'] = st.selectbox('Streaming Movies', options=['Yes', 'No internet service', 'No'])

with col3:
    st.session_state.features['Contract'] = st.selectbox('Current Contract Period', options=['Two year', 'One year', 'Month-to-month'])
    st.session_state.features['Paperless Billing'] = st.selectbox('Paperless Billing', options=['Yes', 'No'])
    st.session_state.features['Payment Method'] = st.selectbox('Payment Method', options=[
        'Credit card (automatic)', 'Mailed check', 'Bank transfer (automatic)', 'Electronic check'
    ])
    st.session_state.features['Monthly Charges'] = st.number_input('Monthly Charges', min_value=0)
    st.session_state.features['Total Charges'] = st.number_input('Total Charges', min_value=st.session_state.features.get('Monthly Charges', 0))
    st.session_state.features['CLTV'] = st.number_input('CLTV', min_value=0)

features = {
    'Senior Citizen': st.session_state.features.get('Senior Citizen', 'No'),
    'Partner': st.session_state.features.get('Partner', 'No'),
    'Dependents': st.session_state.features.get('Dependents', 'No'),
    'Tenure Months': st.session_state.features.get('Tenure Months', 0),
    'Multiple Lines': st.session_state.features.get('Multiple Lines', 'No'),
    'Internet Service': st.session_state.features.get('Internet Service', 'No'),
    'Online Security': 'No internet service' if st.session_state.features.get('Internet Service', 'No') == 'No' else st.session_state.features.get('Online Security', 'No'),
    'Online Backup': 'No internet service' if st.session_state.features.get('Internet Service', 'No') == 'No' else st.session_state.features.get('Online Backup', 'No'),
    'Device Protection': 'No internet service' if st.session_state.features.get('Internet Service', 'No') == 'No' else st.session_state.features.get('Device Protection', 'No'),
    'Tech Support': 'No internet service' if st.session_state.features.get('Internet Service', 'No') == 'No' else st.session_state.features.get('Tech Support', 'No'),
    'Streaming TV': 'No internet service' if st.session_state.features.get('Internet Service', 'No') == 'No' else st.session_state.features.get('Streaming TV', 'No'),
    'Streaming Movies': 'No internet service' if st.session_state.features.get('Internet Service', 'No') == 'No' else st.session_state.features.get('Streaming Movies', 'No'),
    'Contract': st.session_state.features.get('Contract', 'Month-to-month'),
    'Paperless Billing': st.session_state.features.get('Paperless Billing', 'No'),
    'Payment Method': st.session_state.features.get('Payment Method', 'Bank transfer (automatic)'),
    'Monthly Charges': st.session_state.features.get('Monthly Charges', 0),
    'Total Charges': st.session_state.features.get('Total Charges', 0),
    'CLTV': st.session_state.features.get('CLTV', 0)
}

# Test

if st.button('Predict'):
    try:
        predicted_class, predicted_proba = predict(features)
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Prediction Probabilities: {predicted_proba}")

        st.bar_chart(predicted_proba)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
