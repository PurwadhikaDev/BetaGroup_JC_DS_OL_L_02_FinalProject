import streamlit as st
import pandas as pd
import pickle

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title("🛍️ E-Commerce Churn Predictor")
st.markdown("Masukkan data pelanggan sesuai parameter di bawah untuk melihat prediksi.")

# --- 2. MAPPING KATEGORI ---
mapping = {
    "Complain": {"Tidak": 0, "Ya": 1},  # Complain: 0=No Complain, 1=Yes
    "MaritalStatus": {"Divorced": 0, "Married": 1, "Single": 2},  # LabelEncoder urutan alphabetical: Divorced, Married, Single
    "PreferedOrderCat": {"Fashion": 0, "Grocery": 1, "Laptop & Accessory": 2, "Mobile Phone": 3, "Others": 4},  # LabelEncoder urutan alphabetical
    "PreferredPaymentMode": {"Cash on Delivery": 0, "Credit Card": 1, "Debit Card": 2, "E wallet": 3, "UPI": 4},  # LabelEncoder urutan alphabetical
    "Gender": {"Female": 0, "Male": 1},  # LabelEncoder urutan alphabetical
    "PreferredLoginDevice": {"Computer": 0, "Mobile Phone": 1}  # LabelEncoder urutan alphabetical
}

# --- 3. INPUT FORM ---
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (Bulan)", 0.0, 65.0, 10.0)
        complain = st.selectbox("Pernah Komplain?", ["Tidak", "Ya"])
        num_address = st.number_input("Number of Address", 1, 20, 3)
        cashback = st.number_input("Cashback Amount", 0.0, 500.0, 150.0)
        warehouse = st.number_input("Warehouse To Home", 0.0, 150.0, 15.0)
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        marital = st.selectbox("Marital Status", ["Divorced", "Married", "Single"])
        order_cat = st.selectbox("Preferred Order Category", ["Fashion", "Grocery", "Laptop & Accessory", "Mobile Phone", "Others"])

    with col2:
        day_since_last = st.number_input("Day Since Last Order", 0.0, 60.0, 5.0)
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        order_hike = st.number_input("Order Amount Hike (%)", 0.0, 50.0, 15.0)
        order_count = st.number_input("Order Count", 0.0, 30.0, 2.0)
        payment = st.selectbox("Preferred Payment Mode", ["Cash on Delivery", "Credit Card", "Debit Card", "E wallet", "UPI"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        login_device = st.selectbox("Preferred Login Device", ["Computer", "Mobile Phone"])

    submitted = st.form_submit_button("Predict Now")

# --- 4. PREDICTION ---
if submitted:
    # Susun list
    features = [
        tenure,
        mapping["Complain"][complain],
        num_address,
        cashback,
        warehouse,
        satisfaction,
        mapping["MaritalStatus"][marital],
        mapping["PreferedOrderCat"][order_cat],
        day_since_last,
        city_tier,
        order_hike,
        order_count,
        mapping["PreferredPaymentMode"][payment],
        mapping["Gender"][gender],
        mapping["PreferredLoginDevice"][login_device]
    ]

    # Convert ke DataFrame dengan nama kolom yang sesuai dengan training
    feature_names = [
        'Tenure', 'Complain', 'NumberOfAddress', 'CashbackAmount',
        'WarehouseToHome', 'SatisfactionScore', 'MaritalStatus',
        'PreferedOrderCat', 'DaySinceLastOrder', 'CityTier',
        'OrderAmountHikeFromlastYear', 'OrderCount', 'PreferredPaymentMode',
        'Gender', 'PreferredLoginDevice'
    ]
    input_df = pd.DataFrame([features], columns=feature_names)

    # Load dan prepare data training untuk scaler (one-time dengan cache)
    @st.cache_data
    def get_scaler_and_features():
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv('ecommerce_cleaned_data.csv')

        # Pilih fitur sesuai SHAP selection
        selected_features = [
            'Tenure', 'Complain', 'NumberOfAddress', 'CashbackAmount',
            'WarehouseToHome', 'SatisfactionScore', 'MaritalStatus',
            'PreferedOrderCat', 'DaySinceLastOrder', 'CityTier',
            'OrderAmountHikeFromlastYear', 'OrderCount', 'PreferredPaymentMode',
            'Gender', 'PreferredLoginDevice', 'Churn'
        ]

        df_selected = df[selected_features].copy()
        # Manual encoding sesuai mapping di app.py
        df_selected['PreferredLoginDevice'] = df_selected['PreferredLoginDevice'].map({'Computer': 0, 'Mobile Phone': 1})
        df_selected['PreferredPaymentMode'] = df_selected['PreferredPaymentMode'].map({
            'Cash on Delivery': 0, 'Credit Card': 1, 'Debit Card': 2, 'E wallet': 3, 'UPI': 4
        })
        df_selected['Gender'] = df_selected['Gender'].map({'Female': 0, 'Male': 1})
        df_selected['PreferedOrderCat'] = df_selected['PreferedOrderCat'].map({
            'Fashion': 0, 'Grocery': 1, 'Laptop & Accessory': 2, 'Mobile Phone': 3, 'Others': 4
        })
        df_selected['MaritalStatus'] = df_selected['MaritalStatus'].map({'Divorced': 0, 'Married': 1, 'Single': 2})

        # Drop Churn untuk X
        X_train = df_selected.drop('Churn', axis=1)

        # Fit scaler
        scaler_temp = StandardScaler()
        scaler_temp.fit(X_train)

        return scaler_temp

    # Get scaler dari session state
    if 'scaler_fitted' not in st.session_state:
        st.session_state.scaler_fitted = get_scaler_and_features()

    input_scaled = st.session_state.scaler_fitted.transform(input_df)

    # Eksekusi Prediksi
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    if prediction[0] == 1:
        st.error(f"### ⚠️ HASIL: PELANGGAN BERPOTENSI CHURN")
        st.write(f"Tingkat Keyakinan Model: **{probability[1]:.2%}**")
    else:
        st.success(f"### ✅ HASIL: PELANGGAN TETAP SETIA (LOYAL)")

        st.write(f"Tingkat Keyakinan Model: **{probability[0]:.2%}**")
