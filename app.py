import streamlit as st
import pandas as pd
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# Page config
st.set_page_config(
    page_title="NCAA 26 Skill Points Predictor",
    page_icon="🏈",
    layout="centered"
)

# Google Sheets setup
SHEET_ID = "1ANYMLAgjc1nwXYCdm2nbegrgPtUfUGh1aR_nhLxcMK8"
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

# Initialize Google Sheets connection using Streamlit secrets
@st.cache_resource
def get_gsheet_connection():
    """Connect to Google Sheets using secrets"""
    try:
        # Load credentials from Streamlit secrets
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=SCOPES
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SHEET_ID).sheet1
        return sheet
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return None

# Coefficients from Dummy_Model_Clean (506 players, outliers removed)
coefficients = {
    'Intercept': 83.40709,
    'HC_Moti.1': 0.35494,
    'HC_Moti.2': 2.03755,
    'OC_Moti.1': 0.62476,
    'DC_Moti.1': 3.16875,
    'HC_TD1': 9.49064,
    'HC_TD2': 2.77841,
    'HC_TD3': -0.33346,
    'OC_TD1': 3.08600,
    'OC_TD2': 1.82702,
    'OC_TD3': 2.45405,
    'DC_TD1': -1.44690,
    'DC_TD2': 14.21596,
    'DC_TD3': 13.07231,
    'XP_Penalty': -0.42761
}

# Development Trait coefficients (baseline is Elite)
dev_trait_coeffs = {
    'Elite': 0,
    'Impact': -38.18651,
    'Normal': -50.53085,
    'Star': -23.53652
}

# DevT to number mapping for database
dev_trait_num = {
    'Elite': 4,
    'Star': 3,
    'Impact': 2,
    'Normal': 1
}

# Position coefficients (baseline is QB)
position_coeffs = {
    'QB': 0, 'RB': -4.687237, 'WR': -0.3681322, 'TE': -6.699325,
    'OL': 3.123435, 'DL': -6.514467, 'DT': -3.839926, 'LB': -10.68404,
    'S': -2.186808, 'CB': 0.5434067, 'K': 0.2088728, 'P': 0.8526459
}

# Year coefficients (baseline is FR)
year_coeffs = {
    'FR': 0, 'FR (RS)': -2.86779, 'SO': -3.145401, 'SO (RS)': -5.05219,
    'JR': -4.748136, 'JR (RS)': -2.66559, 'SR': 0.08653364
}

# Variable labels
variable_labels = {
    'HC_Moti.1': 'HC Motivator Tier 1',
    'HC_Moti.2': 'HC Motivator Tier 2',
    'OC_Moti.1': 'OC Motivator Tier 1',
    'DC_Moti.1': 'DC Motivator Tier 1',
    'HC_TD1': 'HC Talent Developer Tier 1',
    'HC_TD2': 'HC Talent Developer Tier 2',
    'HC_TD3': 'HC Talent Developer Tier 3',
    'OC_TD1': 'OC Talent Developer Tier 1',
    'OC_TD2': 'OC Talent Developer Tier 2',
    'OC_TD3': 'OC Talent Developer Tier 3',
    'DC_TD1': 'DC Talent Developer Tier 1',
    'DC_TD2': 'DC Talent Developer Tier 2',
    'DC_TD3': 'DC Talent Developer Tier 3'
}

# Model performance statistics
MODEL_STATS = {
    'r_squared': 0.9322,
    'adj_r_squared': 0.9273,
    'mae': 4.39,
    'rmse': 5.65,
    'n': 506
}

# DevT-specific accuracy data
DEVT_ACCURACY = {
    'Elite': {
        'n': 11, 'mae': 8.62,
        'ranges': [
            {'range': 5, 'percentage': 27.3},
            {'range': 10, 'percentage': 63.6},
            {'range': 15, 'percentage': 90.9}
        ]
    },
    'Impact': {
        'n': 250, 'mae': 4.05,
        'ranges': [
            {'range': 5, 'percentage': 68.8},
            {'range': 10, 'percentage': 94.8},
            {'range': 15, 'percentage': 99.2}
        ]
    },
    'Normal': {
        'n': 116, 'mae': 5.5,
        'ranges': [
            {'range': 5, 'percentage': 51.7},
            {'range': 10, 'percentage': 82.8},
            {'range': 15, 'percentage': 99.1}
        ]
    },
    'Star': {
        'n': 129, 'mae': 5.39,
        'ranges': [
            {'range': 5, 'percentage': 48.1},
            {'range': 10, 'percentage': 86.8},
            {'range': 15, 'percentage': 97.7}
        ]
    }
}

# Database functions
def save_complete_data(prediction_data, actual_points):
    """Save complete data (prediction + actual) to Google Sheets"""
    try:
        sheet = get_gsheet_connection()
        if sheet is None:
            return False
        
        row = [
            prediction_data['team'],
            prediction_data['player_name'],
            actual_points,
            prediction_data['position'],
            prediction_data['year'],
            prediction_data['dev_trait'],
            dev_trait_num[prediction_data['dev_trait']],
            prediction_data['snaps'],
            prediction_data['HC_Moti.1'],
            prediction_data['HC_Moti.2'],
            prediction_data['OC_Moti.1'],
            prediction_data['DC_Moti.1'],
            prediction_data['HC_TD1'],
            prediction_data['HC_TD2'],
            prediction_data['HC_TD3'],
            prediction_data['OC_TD1'],
            prediction_data['OC_TD2'],
            prediction_data['OC_TD3'],
            prediction_data['DC_TD1'],
            prediction_data['DC_TD2'],
            prediction_data['DC_TD3'],
            prediction_data['xp_penalty']
        ]
        
        sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Error saving to database: {e}")
        return False

def calculate_prediction(position, year, dev_trait, xp_penalty, coaching_abilities):
    """Calculate skill points prediction"""
    prediction = coefficients['Intercept']
    prediction += position_coeffs[position]
    prediction += year_coeffs[year]
    prediction += dev_trait_coeffs[dev_trait]
    prediction += coefficients['XP_Penalty'] * xp_penalty
    
    for var, value in coaching_abilities.items():
        if value:
            prediction += coefficients[var]
    
    return prediction

def main():
    st.title("🏈 NCAA 26 Skill Points Predictor")
    st.caption(f"v2.3 | Clean Model (R² = {MODEL_STATS['r_squared']:.4f}, MAE = {MODEL_STATS['mae']:.2f})")
    
    st.markdown("---")
    
    st.subheader("Player Information")
    
    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Team", placeholder="e.g., Auburn")
    with col2:
        player_name = st.text_input("Player Name", placeholder="e.g., John Smith")
    
    col1, col2 = st.columns(2)
    
    with col1:
        position = st.selectbox("Position", list(position_coeffs.keys()), index=0)
        year = st.selectbox("Year", list(year_coeffs.keys()), index=0)
    
    with col2:
        dev_trait = st.selectbox("Development Trait", list(dev_trait_coeffs.keys()), index=2)
        xp_penalty = st.number_input("XP Penalty Slider", min_value=0, max_value=100, value=0, step=1)
    
    snaps = st.number_input("Snaps Played", min_value=0, max_value=2000, value=0, step=1)
    
    st.markdown("---")
    st.subheader("Coach Abilities")
    
    coaching_abilities = {}
    
    col1, col2 = st.columns(2)
    
    items = list(variable_labels.items())
    mid = len(items) // 2
    
    with col1:
        for var, label in items[:mid]:
            coaching_abilities[var] = st.checkbox(label, value=False, key=var)
    
    with col2:
        for var, label in items[mid:]:
            coaching_abilities[var] = st.checkbox(label, value=False, key=var)
    
    st.markdown("---")
    
    if st.button("🎯 Predict Skill Points", type="primary", use_container_width=True):
        prediction = calculate_prediction(position, year, dev_trait, xp_penalty, coaching_abilities)
        
        st.session_state.last_prediction = prediction
        st.session_state.last_dev_trait = dev_trait
        st.session_state.last_inputs = {
            'team': team_name,
            'player_name': player_name,
            'snaps': snaps,
            'position': position,
            'year': year,
            'dev_trait': dev_trait,
            'xp_penalty': xp_penalty,
            'HC_Moti.1': 1 if coaching_abilities['HC_Moti.1'] else 0,
            'HC_Moti.2': 1 if coaching_abilities['HC_Moti.2'] else 0,
            'OC_Moti.1': 1 if coaching_abilities['OC_Moti.1'] else 0,
            'DC_Moti.1': 1 if coaching_abilities['DC_Moti.1'] else 0,
            'HC_TD1': 1 if coaching_abilities['HC_TD1'] else 0,
            'HC_TD2': 1 if coaching_abilities['HC_TD2'] else 0,
            'HC_TD3': 1 if coaching_abilities['HC_TD3'] else 0,
            'OC_TD1': 1 if coaching_abilities['OC_TD1'] else 0,
            'OC_TD2': 1 if coaching_abilities['OC_TD2'] else 0,
            'OC_TD3': 1 if coaching_abilities['OC_TD3'] else 0,
            'DC_TD1': 1 if coaching_abilities['DC_TD1'] else 0,
            'DC_TD2': 1 if coaching_abilities['DC_TD2'] else 0,
            'DC_TD3': 1 if coaching_abilities['DC_TD3'] else 0
        }
    
    if 'last_prediction' in st.session_state:
        prediction = st.session_state.last_prediction
        dev_trait = st.session_state.last_dev_trait
        
        st.success(f"### Predicted: {prediction:.1f} skill points")
        
        devt_stats = DEVT_ACCURACY[dev_trait]
        
        st.info(f"""
        **Accuracy for {dev_trait} players** (based on {devt_stats['n']} players)  
        Typical error: ±{devt_stats['mae']:.1f} points
        """)
        
        for acc in devt_stats['ranges']:
            range_val = acc['range']
            pct = acc['percentage']
            lower = max(0, prediction - range_val)
            upper = prediction + range_val
            st.write(f"±{int(range_val)} points ({pct:.1f}% of the time): **{lower:.1f} - {upper:.1f}**")
        
        st.markdown("---")
        
        with st.expander("📊 Help improve the model - Submit actual results"):
            st.write("After checking your Training Results screen, come back and enter the actual skill points!")
            
            actual_points = st.number_input(
                "Actual Skill Points from Training Results:",
                min_value=0,
                max_value=200,
                value=int(prediction),
                step=1,
                key="actual_points_input"
            )
            
            if st.button("Submit Actual Results", key="submit_actual"):
                error = abs(actual_points - prediction)
                
                if save_complete_data(st.session_state.last_inputs, actual_points):
                    st.success(f"✅ Thank you! Data saved to database. Prediction error was {error:.1f} points")
                    st.balloons()
                else:
                    st.error("Could not save to database")
    
    st.markdown("---")
    st.caption("59% of predictions within ±5 points | 89% within ±10 points")
    st.caption("Created by Alex Swanner")

if __name__ == "__main__":
    main()
