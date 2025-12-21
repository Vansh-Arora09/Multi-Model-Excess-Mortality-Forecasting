import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_assets():
    
    try:
        
        rfc_final = joblib.load('rfc_tuned.joblib')
        sc_pop = joblib.load('scaler_pop.joblib') 
        sc_year = joblib.load('scaler_year.joblib') 
        sc_pca = joblib.load('scaler_pca.joblib') 
        pca = joblib.load('pca_object.joblib') 
        lb = joblib.load('label_encoder.joblib') 

    except FileNotFoundError:
        st.error("Model assets not found! Please ensure you have run the joblib_dump.py script and placed the 6 .joblib files in the same directory as app.py.")
        st.stop()
        
    OHE_TRAINING_COLUMNS = [
        'PC1', 'PC2', 'Population_scaled_log', 'HHS Region_1', 'HHS Region_10', 'HHS Region_2', 'HHS Region_3', 'HHS Region_4', 'HHS Region_5', 'HHS Region_6', 'HHS Region_7', 'HHS Region_8', 'HHS Region_9', 'Benchmark_2010 Fixed', 'Benchmark_Floating', 'Age Range_0-54', 'Age Range_0-59', 'Age Range_0-64', 'Age Range_0-69', 'Age Range_0-74', 'Age Range_0-79', 'Age Range_0-84', 'Locality_Metropolitan', 'Locality_Nonmetropolitan', 'Year_Encoded'
    ]
    
    MEDIANS_TRAIN = {'Population': 1500000}
    
    return rfc_final, sc_pop, sc_year, sc_pca, pca, lb, OHE_TRAINING_COLUMNS, MEDIANS_TRAIN


rfc_final, sc_pop, sc_year, sc_pca, pca, lb, OHE_TRAINING_COLUMNS, MEDIANS_TRAIN = load_assets()

def predictor_pipeline(raw_input_df):
    
    data = raw_input_df.copy()
    
    data = data.fillna(MEDIANS_TRAIN)
    data['HHS Region'] = data['HHS Region'].astype(str)

    
    mortality_features = ['Observed Deaths', 'Expected Deaths', 'Potentially Excess Deaths', 'Percent Potentially Excess Deaths']
    
    mortality_scaled = sc_pca.transform(data[mortality_features])
    
    principal_components = pca.transform(mortality_scaled)
    
    data['PC1'] = principal_components[:, 0]
    data['PC2'] = principal_components[:, 1]
    
    data['Population_log'] = np.log1p(data['Population'])
    data['Population_scaled_log'] = sc_pop.transform(data[['Population_log']])
    
    
    data['Year_Encoded'] = sc_year.transform(data[['Year']])
    
    
    cols_for_ohe = ['HHS Region', 'Age Range', 'Benchmark', 'Locality']
    
    data_encoded = pd.get_dummies(data, columns=cols_for_ohe, drop_first=False)

    cols_to_drop = mortality_features + ['Population', 'Population_log', 'Year']
    data_final = data_encoded.drop(columns=cols_to_drop, errors='ignore')
    
    X_final = data_final.reindex(columns=OHE_TRAINING_COLUMNS, fill_value=0)
    
    prediction_proba = rfc_final.predict_proba(X_final)[0]
    
    # 2. Get the Top 1 (Your original return values)
    prediction_encoded = prediction_proba.argmax()
    predicted_label = lb.inverse_transform([prediction_encoded])[0]
    confidence = prediction_proba.max()
    
    # 3. Create the Top 3 Dataframe for the chart
    prob_df = pd.DataFrame({
        'Cause': lb.classes_,
        'Probability': prediction_proba
    }).sort_values(by='Probability', ascending=False).head(3)
    
    return predicted_label, confidence, prob_df
    # --- NEW LOGIC END ---

import plotly.graph_objects as go

def risk_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%", 'font': {'size': 26, 'color': "#0369A1", 'family': "Arial"}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#0369A1"},
            'bar': {'color': "#0369A1"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#7DD3FC",
            'steps': [
                {'range': [0, 50], 'color': '#F87171'},   # Red
                {'range': [50, 80], 'color': '#FBBF24'},  # Yellow
                {'range': [80, 100], 'color': '#34D399'}  # Green
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

import plotly.express as px

def probability_chart(top_3_df):
    df = top_3_df.sort_values(by='Probability', ascending=True)
    
    # Define the colors
    # Winner: Emerald, Runners up: Sky Blue
    bar_colors = ['#0369A1'] * len(df) 
    bar_colors[-1] = '#059669' 
    
    # Border colors (The "Shading"): Deep Blue and Deep Green
    border_colors = ['#0369A1'] * len(df)
    border_colors[-1] = '#059669'

    fig = px.bar(
        df, 
        x='Probability', 
        y='Cause', 
        orientation='h',
        text_auto='.1%',
    )
    
    fig.update_traces(
        marker_color=bar_colors,
        marker_line_color=border_colors, # Dark "shading" border
        marker_line_width=2.5,           # Thick enough to look like a shadow/frame
        opacity=1.0,
        width=0.7,                       # Slightly thicker bars
        # Shallow rounding (reduced from 10 to 4)
        marker_cornerradius=9,           
        # Formatting the Percentage text inside/outside bars
        textfont=dict(size=14, color="#FFFFFF", family="Arial Black"), 
        textposition='auto'
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(
            showgrid=False, 
            # Bolder text for the Cause names
            tickfont=dict(size=14, color="#034977", family="Arial Black"),
            title=""
        )
    )
    
    return fig



def set_custom_style():
    st.markdown("""
        <style>
        /* Main page background - Dark Glowing Black */
        .stApp {
            background-color: #001219; 
        }
        
        /* Sidebar styling - Light Blue */
        [data-testid="stSidebar"] {
            background-color: #BAE6FD; 
            padding: 20px;
            border-radius: 10px;
        }

        /* FIX 1: Force Sidebar Headings to Rich Black */
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] .stMarkdown p {
            color: #001219 !important;
            font-weight: bold !important;
        }

        /* FIX 2: Force Sidebar Input Labels (text above boxes) to Rich Black */
        [data-testid="stSidebar"] label p {
            color: #001219 !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
        }
        /* 1. The Standard State of the Button */
        [data-testid="stSidebar"] .stButton>button {
            width: 100%;
            border-radius: 8px;
            background-color: #059669; /* Emerald Green */
            color: white !important;
            font-size: 1.1em;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease; /* Smooth transition */
        }

        /* 2. THE FIX: The Hover State */
        [data-testid="stSidebar"] .stButton>button:hover {
            background-color: #10B981 !important; /* Brighter Green on hover */
            color: #FFFFFF !important;
            border: 1px solid #FFFFFF;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.6); /* Glowing effect */
            transform: translateY(-2px); /* Slight lift effect */
        }

        /* 3. The Active/Click State */
        [data-testid="stSidebar"] .stButton>button:active {
            background-color: #047857 !important;
            transform: translateY(0px);
        }
        /* FIX 3: Force Sidebar Helper text and subheaders */
        [data-testid="stSidebar"] .stMarkdown h2 {
            border-bottom: 2px solid #0369A1;
            padding-bottom: 5px;
        }

        .ecg-container {
            position: relative;
            width: 100%;
            height: 60px;
            background-color: transparent;
            overflow: hidden;
            margin-bottom: -40px;
        }

        /* Title and Header formatting for Main Content */
        .stApp h1, .stApp h2, .stApp h3, .stApp p {
            color: #ECFDF5; 
            font-family: 'Georgia', serif;
        }
        
        /* Result box */
        .result-box {
            display: table;
            margin: 55px auto;
            padding: 15px 40px;
            border-radius: 15px;
            background-color: #F0F9FF; 
            color: #0369A1; 
            text-align: center;
            font-size: 1.8em;
            font-weight: bold;
            border: 3px solid #7DD3FC;
            box-shadow: 0 0 15px rgba(125, 211, 252, 0.7);
            animation: pulse 2.5s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 5px rgba(125, 211, 252, 0.4); transform: scale(1); }
            50% { box-shadow: 0 0 25px rgba(56, 189, 248, 0.8); transform: scale(1.02); }
            100% { box-shadow: 0 0 5px rgba(125, 211, 252, 0.4); transform: scale(1); }
        }
        
        /* Heartbeat Animation */
        .heartbeat-line {
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, transparent, #059669, transparent);
            position: absolute;
            bottom: 0;
            animation: heartbeat 1.5s infinite linear;
        }

        @keyframes heartbeat {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

set_custom_style()


AGE_RANGES = ['0-54', '0-59', '0-64', '0-69', '0-74', '0-79', '0-84'] 
LOCALITIES = ['Metropolitan', 'Nonmetropolitan']
BENCHMARKS = ['2010 Fixed', 'Floating'] 
HHS_REGIONS = [str(i) for i in range(1, 11)] 


st.title("PulsePredict 🩺")
st.markdown("### *Mapping mortality trends with predictive intelligence*")
st.markdown("---")



st.sidebar.header("1. Core Metadata 🔑")

year = st.sidebar.number_input("Year (Trained on 2005-2015)", min_value=2005, max_value=2030, value=2015, step=1, help="The year for which the prediction is made.")
population = st.sidebar.number_input("Population", min_value=1000.0, max_value=20000000.0, value=2500000.0, step=10000.0, format="%.1f", help="The population of the area.")


hhs_region = st.sidebar.selectbox("HHS Region", options=HHS_REGIONS, help="The Health and Human Services region (1-10).")
age_range = st.sidebar.selectbox("Age Range (Matching Model Feature)", options=AGE_RANGES)
locality = st.sidebar.selectbox("Locality Type", options=LOCALITIES)
benchmark = st.sidebar.selectbox("Benchmark", options=BENCHMARKS)


st.sidebar.header("2. Mortality Biomarkers 🏥")
st.sidebar.markdown("_Processes inputs to determine core drivers ($PC1$ & $PC2$)._")


observed_deaths = st.sidebar.number_input("Observed Deaths", min_value=0.0, max_value=500000.0, value=1500.0, format="%.1f")
expected_deaths = st.sidebar.number_input("Expected Deaths", min_value=0.0, max_value=500000.0, value=1000.0, format="%.1f")
pot_excess_deaths = st.sidebar.number_input("Potentially Excess Deaths", min_value=0.0, max_value=200000.0, value=500.0, format="%.1f")
percent_excess = st.sidebar.number_input("Percent Potentially Excess Deaths (%)", min_value=0.0, max_value=100.0, value=50.0, format="%.1f")


input_data = pd.DataFrame([{
    'Year': year, 
    'HHS Region': hhs_region, 
    'Age Range': age_range,
    'Benchmark': benchmark,
    'Locality': locality,
    'Observed Deaths': observed_deaths, 
    'Population': population,
    'Expected Deaths': expected_deaths, 
    'Potentially Excess Deaths': pot_excess_deaths,
    'Percent Potentially Excess Deaths': percent_excess
}])

st.header("🔍 Diagnostic Verdict")
if st.sidebar.button('**Generate Prediction**'):
   
    if observed_deaths < expected_deaths and pot_excess_deaths > 0:
         st.warning("⚠️ **Data Inconsistency Detected:** Observed deaths are lower than expected, yet excess deaths are positive.")
    
    with st.spinner('🧠 PulsePredict AI is analyzing health drivers...'):
        try:
            # 1. Get the 3 return values from your updated pipeline
            predicted_cause, confidence, top_3_df = predictor_pipeline(input_data)
            
            # 2. Top Section: Gauge and Result Box
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                # Gauge Chart (Risk Meter)
                st.plotly_chart(risk_gauge(confidence), use_container_width=True)
            
            with res_col2:
                
                #st.markdown("### Forecasted Primary Cause")
                # ECG Heartbeat Line + Animated Result Box
                st.markdown("""
                    <div class="ecg-container">
                        <div class="heartbeat-line"></div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(
                    f'<div class="result-box">🚨 {predicted_cause}</div>', 
                    unsafe_allow_html=True
                )
                
                # Confidence Metric
                #st.metric(label="Model Certainty", value=f"{confidence:.1%}")

            # 3. Middle Section: Distribution Chart
            st.markdown("---")
            st.subheader("📊 Primary Risk Distribution")

            # Display the new sleek Plotly chart
            st.plotly_chart(probability_chart(top_3_df), use_container_width=True)
            importances = rfc_final.feature_importances_
            feat_importances = pd.Series(importances, index=OHE_TRAINING_COLUMNS)
            top_contributor = feat_importances.idxmax()

            st.info(f"🔍 **Primary Driver:** The model's decision was most influenced by **{top_contributor}**.")
            # 4. Confidence Messaging
            if confidence > 0.80:
                st.success("✅ **High Confidence:** The model is very certain about this trend.")
            elif confidence > 0.50:
                st.info("ℹ️ **Moderate Confidence:** This is the most likely cause.")
            else:
                st.warning("⚠️ **Low Confidence:** Interpret this result with caution.")

            # 5. Bottom Section: Data Audit
            with st.expander("🔍 View Feature Analysis"):
                st.markdown('<p class="audit-header">Patient Health Driver Audit</p>', unsafe_allow_html=True)
                st.write("Detailed breakdown of inputs processed by PulsePredict AI:")
                
                # We style the dataframe columns to have a specific color theme
                styled_df = input_data.T.style.set_properties(**{
                    'background-color': '#F0F9FF',
                    'color': '#001219',
                    'border-color': '#BAE6FD',
                    'font-family': 'Verdana'
                }).set_table_styles([
                    {'selector': 'th', 'props': [('background-color', '#0369A1'), ('color', 'white'), ('font-weight', 'bold')]}
                ])

                st.dataframe(styled_df, use_container_width=True)
                
                st.info("💡 **Note:** These values have been normalized and validated against the 2024 health driver baseline.") 
                        
        except Exception as e:
            st.error(f"❌ **Prediction Error:** {e}")
            st.info("💡 **Troubleshooting:** Ensure all 6 `.joblib` files are loaded.")