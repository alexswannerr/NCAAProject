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

# ===== UPDATED COEFFICIENTS (899 entries, Season 1-4, QB Baseline) =====
coefficients = {
    'Intercept': 81.4297,
    'HC_Moti.1': 0.5231,
    'HC_Moti.2': 1.1197,
    'OC_Moti.1': 1.7403,
    'DC_Moti.1': 1.5185,
    'HC_TD1': 6.6249,
    'HC_TD2': 2.7500,
    'HC_TD3': -1.5018,
    'OC_TD1': 2.1065,
    'OC_TD2': 2.1532,
    'OC_TD3': 2.5299,
    'DC_TD1': 4.6967,
    'DC_TD2': 1.8046,
    'DC_TD3': -0.8955,
    'XP_Penalty': -0.4659
}

# Development Trait coefficients (baseline is Elite)
dev_trait_coeffs = {
    'Elite': 0,
    'Star': -21.0859,
    'Impact': -33.9941,
    'Normal': -45.7699
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
    'CB': -0.3893,
    'DL': -5.8792,
    'K': 0.3973,
    'LB': -9.8241,
    'OL': 1.7122,
    'P': 0.8673,
    'RB': -4.9690,
    'S': -2.8760,
    'TE': -6.4549,
    'WR': -1.3104
}

# Year coefficients (baseline is FR)
year_coeffs = {
    'FR': 0,
    'FR (RS)': -2.2868,
    'SO': -3.2333,
    'SO (RS)': -4.5611,
    'JR': -4.2671,
    'JR (RS)': -1.9461,
    'SR': -4.2671,  # Using JR as proxy
    'SR (RS)': -1.9461  # Using JR(RS) as proxy
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
    'r_squared': 0.91208,
    'adj_r_squared': 0.91208,
    'mae': 4.75,
    'rmse': 6.25,
    'n': 899
}

# ===== UPDATED DevT-specific accuracy data =====
DEVT_ACCURACY = {
    'Elite': {
        'n': 37,
        'mae': 8.90,
        'ranges': [
            {'range': 5, 'percentage': 27.0},
            {'range': 10, 'percentage': 64.9},
            {'range': 15, 'percentage': 83.8}
        ]
    },
    'Star': {
        'n': 255,
        'mae': 5.95,
        'ranges': [
            {'range': 5, 'percentage': 49.8},
            {'range': 10, 'percentage': 82.7},
            {'range': 15, 'percentage': 95.3}
        ]
    },
    'Impact': {
        'n': 411,
        'mae': 3.14,
        'ranges': [
            {'range': 5, 'percentage': 82.7},
            {'range': 10, 'percentage': 98.3},
            {'range': 15, 'percentage': 99.8}
        ]
    },
    'Normal': {
        'n': 196,
        'mae': 5.67,
        'ranges': [
            {'range': 5, 'percentage': 48.0},
            {'range': 10, 'percentage': 86.7},
            {'range': 15, 'percentage': 96.4}
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
    st.caption(f"v4.0 | Updated Model (R² = {MODEL_STATS['r_squared']:.5f}, MAE = {MODEL_STATS['mae']:.2f})")
    
    st.markdown("---")
    
    st.subheader("Player Information")
    
    col1, col2 = st.columns(2)
    with col1:
        team_name = st.text_input("Team", placeholder="e.g., Ohio State")
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
    st.caption("64% of predictions within ±5 points | 90% within ±10 points | 99% within ±20 points")
    st.caption("Model trained on 899 players (seasons 1-4) | Best for Impact players (98% ±10 accuracy)")
    st.caption("Created by Alex Swanner | [LinkedIn](https://linkedin.com/in/alexswanner/)")

if __name__ == "__main__":
    main()
