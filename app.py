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

# ===== UPDATED COEFFICIENTS (610 entries, No Auburn) =====
coefficients = {
    'Intercept': 84.3846,
    'HC_Moti.1': 0.8413,
    'HC_Moti.2': 1.3696,
    'OC_Moti.1': 0.1399,
    'DC_Moti.1': 2.0169,
    'HC_TD1': 6.3471,
    'HC_TD2': 1.5988,
    'HC_TD3': -2.6348,
    'OC_TD1': 0.9023,
    'OC_TD2': 0.9902,
    'OC_TD3': 4.0082,
    'DC_TD1': 3.5946,
    'DC_TD2': 1.8137,
    'DC_TD3': -1.0978,
    'XP_Penalty': -0.4372
}

# Development Trait coefficients (baseline is Elite)
dev_trait_coeffs = {
    'Elite': 0,
    'Star': -23.4378,
    'Impact': -36.3314,
    'Normal': -47.7393
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
    'QB': 0,
    'RB': -6.2637,
    'WR': -2.4537,
    'TE': -7.1273,
    'OL': -0.1576,
    'DL': -8.0265,
    'LB': -11.7211,
    'CB': -1.7674,
    'S': -4.449,
    'K': -1.98,
    'P': -2.2867
}

# Year coefficients (baseline is FR)
year_coeffs = {
    'FR': 0,
    'FR (RS)': -3.7344,
    'SO': -1.8989,
    'SO (RS)': -6.2791,
    'JR': -3.9333,
    'JR (RS)': -3.0805
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

# ===== UPDATED MODEL PERFORMANCE STATISTICS =====
MODEL_STATS = {
    'r_squared': 0.92155,
    'adj_r_squared': 0.92155,
    'mae': 3.991,
    'rmse': 5.209,
    'n': 610
}

# ===== UPDATED DevT-specific accuracy data =====
DEVT_ACCURACY = {
    'Elite': {
        'n': 8,
        'mae': 8.26,
        'ranges': [
            {'range': 5, 'percentage': 37.5},
            {'range': 10, 'percentage': 50.0},
            {'range': 15, 'percentage': 75.0}
        ]
    },
    'Star': {
        'n': 145,
        'mae': 5.60,
        'ranges': [
            {'range': 5, 'percentage': 51.0},
            {'range': 10, 'percentage': 86.9},
            {'range': 15, 'percentage': 96.6}
        ]
    },
    'Impact': {
        'n': 300,
        'mae': 3.18,
        'ranges': [
            {'range': 5, 'percentage': 72.3},
            {'range': 10, 'percentage': 98.3},
            {'range': 15, 'percentage': 99.7}
        ]
    },
    'Normal': {
        'n': 157,
        'mae': 4.43,
        'ranges': [
            {'range': 5, 'percentage': 62.4},
            {'range': 10, 'percentage': 93.6},
            {'range': 15, 'percentage': 98.7}
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
    """Calculate skill points prediction with floor constraint"""
    prediction = coefficients['Intercept']
    prediction += position_coeffs[position]
    prediction += year_coeffs[year]
    prediction += dev_trait_coeffs[dev_trait]
    prediction += coefficients['XP_Penalty'] * xp_penalty
    
    for var, value in coaching_abilities.items():
        if value:
            prediction += coefficients[var]
    
    # Apply floor constraint (no negative predictions)
    prediction = max(0, prediction)
    
    return prediction

def main():
    st.title("🏈 NCAA 26 Skill Points Predictor")
    st.caption(f"v3.0 | Updated Model (R² = {MODEL_STATS['r_squared']:.5f}, MAE = {MODEL_STATS['mae']:.2f})")
    
    st.markdown("---")
    
    st.subheader("Player Information")
    
    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Team", placeholder="e.g., Georgia")
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
        Typical error: ±{devt_stats['mae']:.2f} points
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
    st.caption("71% of predictions within ±5 points | 94% within ±10 points | 99% within ±15 points")
    st.caption("Model trained on 610 players (seasons 1-5) | Optimized for early-mid dynasty")
    st.caption("Created by Alex Swanner | [LinkedIn](https://linkedin.com/in/alexswanner/)")

if __name__ == "__main__":
    main()
