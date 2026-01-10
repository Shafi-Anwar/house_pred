import streamlit as st
import requests
import json
import time
st.set_page_config(
    page_title="",
    page_icon="🏡",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton > button {
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4);
    }
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    .feature-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #4F46E5;
        margin: 0.5rem 0;
        color:black
    }
    .api-status {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .api-online { background: #10b981; color: white; }
    .api-offline { background: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)

st.title("🏡 California House Price Predictor")
st.caption("Powered by Random Forest with Feature Engineering • R² Score: 0.84")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_url = st.text_input(
        "API Endpoint",
        value="http://127.0.0.1:5000/predict",
        help="URL of your Flask prediction API"
    )
    
    st.divider()
    
    st.markdown("###  Model Info")
    st.markdown("""
    **Model:** RandomForestRegressor  
    **Features:** 13 total  
    **Transform:** Log1p + Expm1  
    **Target:** House Price ($100k units)
    """)
    
    st.divider()
    
    if st.button("Test API Connection"):
        with st.spinner("Testing connection..."):
            try:
                response = requests.get(api_url.replace("/predict", "/health"), timeout=5)
                if response.status_code == 200:
                    st.success("API is online!")
                else:
                    st.error("API responded with error")
            except:
                st.error("Cannot connect to API")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 📝 Enter House Features")
with col2:
    try:
        response = requests.get(api_url.replace("/predict", "/health"), timeout=2)
        status = "online" if response.status_code == 200 else "offline"
        st.markdown(f'<span class="api-status api-{status}">API {status.upper()}</span>', 
                   unsafe_allow_html=True)
    except:
        st.markdown('<span class="api-status api-offline">API OFFLINE</span>', 
                   unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="feature-card">Location</div>', unsafe_allow_html=True)
    Latitude = st.slider(
        "Latitude", 
        min_value=32.0, 
        max_value=42.0, 
        value=34.05,
        help="Northern California (42) to Southern California (32)"
    )
    Longitude = st.slider(
        "Longitude", 
        min_value=-124.0, 
        max_value=-114.0, 
        value=-118.25,
        help="Coastal (-124) to Inland (-114)"
    )
    
    st.markdown('<div class="feature-card">House Features</div>', unsafe_allow_html=True)
    HouseAge = st.slider("House Age (years)", 1, 100, 25, help="Age of the house")
    AveRooms = st.slider("Average Rooms", 1.0, 20.0, 6.0, step=0.5, help="Average rooms per household")
    AveBedrms = st.slider("Average Bedrooms", 0.5, 10.0, 1.2, step=0.1, help="Average bedrooms per household")

with col2:
    st.markdown('<div class="feature-card">Income & Population</div>', unsafe_allow_html=True)
    MedInc = st.slider(
        "Median Income (scaled)", 
        0.5, 15.0, 5.0, step=0.1,
        help="Median income in the area (scaled to $100,000s)"
    )
    Population = st.slider(
        "Population", 
        100, 10000, 1200, step=50,
        help="Total population in the block"
    )
    AveOccup = st.slider(
        "Average Occupancy", 
        1.0, 10.0, 3.0, step=0.1,
        help="Average number of household members"
    )
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button(
        "🚀 Predict House Price", 
        type="primary",
        use_container_width=True
    )
if predict_btn:
    payload = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }

    with st.spinner(" Calling ML API..."):
        time.sleep(0.5) 
        
        try:
            response = requests.post(api_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "prediction" in result and "formatted" in result["prediction"]:
                    price_display = result["prediction"]["formatted"]
                    price_value = result["prediction"]["us_dollars"]
                elif "predicted_price" in result:
                    price_display = f"${float(result['predicted_price']):,.2f}"
                    price_value = float(result['predicted_price'])
                else:
                    price_display = f"${result.get('predicted_price', 0):,.2f}"
                    price_value = result.get('predicted_price', 0)
                st.markdown(f"""
                <div class="price-card">
                    <h2 style="margin:0; font-size: 2.5rem;">{price_display}</h2>
                    <p style="opacity: 0.9; margin-top: 0.5rem;">Predicted Market Value</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.progress(0.84, text=f"Model Confidence (R² Score: 84%)")
                
                eng_col1, eng_col2 = st.columns(2)
                
                with st.expander("View API Response Details"):
                    st.json(result)
                    
            elif response.status_code == 400:
                error_data = response.json()
                st.error(f" Input Error: {error_data.get('error', 'Invalid data')}")
                if 'missing' in error_data:
                    st.warning(f"Missing fields: {', '.join(error_data['missing'])}")
                    
            elif response.status_code == 500:
                st.error(" API Server Error - Check if model is loaded")
                try:
                    error_data = response.json()
                    st.code(json.dumps(error_data, indent=2))
                except:
                    st.code(response.text)
                    
            else:
                st.error(f" Unexpected Error (Status: {response.status_code})")
                st.code(response.text)
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Connection Failed - Make sure Flask API is running!")
            st.info(f"Your API should be running at: `{api_url}`")
            st.code("python app.py  # Run this in your terminal")
            
        except requests.exceptions.Timeout:
            st.error("Request Timeout - API is not responding")
            
        except Exception as e:
            st.error(f" Unexpected Error: {str(e)}")
st.divider()

with st.expander("ℹ About This Model"):
    st.markdown("""
    ### 🏆 Model Highlights
    
    **Feature Engineering:**
    - **Income per Occupant**: Most important feature
    - **Distance to Coast**: Geographic premium
    - **Room/Bedroom Ratios**: Space utilization
    - **Population Density**: Neighborhood context
    
    **Technical Specs:**
    - Algorithm: Random Forest Regressor
    - R² Score: 0.84 (84% variance explained)
    - Features: 13 total (8 original + 5 engineered)
    - Transformation: Log1p → Predict → Expm1
    
    **API Endpoints:**
    - `POST /predict` - Make predictions
    - `GET /health` - API status check
    - `GET /features` - List of all features
    """)

st.caption("Built with 🧠 using Streamlit + Flask | Model: RandomForest (R² 0.84) | Shafi - Data scientist/ML Engineer")