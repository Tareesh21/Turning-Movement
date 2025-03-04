import streamlit as st
from google.cloud import bigquery
import pandas as pd
import plotly.express as px
from google.oauth2 import service_account

# ✅ Set Streamlit Page Config
st.set_page_config(page_title="Traffic Movement Analysis Dashboard", layout="wide")

# ✅ Load BigQuery Credentials
try:
    service_account_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    client = bigquery.Client(credentials=credentials, project=service_account_info["project_id"])
except Exception as e:
    st.error(f"❌ Error loading Google Cloud credentials: {e}")
    st.stop()  # Stop execution if authentication fails

# ✅ Fetch Data from BigQuery
try:
    query = """
    SELECT * FROM `proj-452520.TMC.TurningMC`
    LIMIT 1000
    """
    df = client.query(query).to_dataframe()
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# ✅ Data Cleaning & Preprocessing
df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

df_cleaned = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE'])

# ✅ Sidebar Filters
st.sidebar.header("🔍 Filter Data")
df_cleaned = df_cleaned.dropna(subset=['DATE'])
default_start, default_end = df_cleaned['DATE'].min(), df_cleaned['DATE'].max()

date_range = st.sidebar.date_input("Select Date Range", [default_start, default_end])
filtered_df = df_cleaned[
    (df_cleaned['DATE'].dt.date >= date_range[0]) & 
    (df_cleaned['DATE'].dt.date <= date_range[1])
]

unique_intersections = df_cleaned["INTNAME"].dropna().unique().tolist()
selected_intersection = st.sidebar.selectbox("Select Intersection", ["All"] + unique_intersections)

if selected_intersection != "All":
    filtered_df = filtered_df[filtered_df["INTNAME"] == selected_intersection]

# ✅ Dashboard Visualizations
st.title("🚦 Traffic Movement Analysis")

st.subheader("📊 Top 10 Intersections with Highest Traffic Volume")
top_intersections = (
    filtered_df.groupby('INTNAME')['AUTONBL']
    .sum()
    .reset_index()
    .sort_values(by='AUTONBL', ascending=False)
    .head(10)
)

fig_bar = px.bar(top_intersections, x="INTNAME", y="AUTONBL",
                 labels={"INTNAME": "Intersection", "AUTONBL": "Vehicle Count"},
                 title="Top 10 Intersections by Traffic Volume")
st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("📈 Traffic Volume Trends Over Time")
df_time_series = filtered_df.groupby('DATE')['AUTONBL'].sum().reset_index()

fig_line = px.line(df_time_series, x='DATE', y='AUTONBL',
                   labels={'AUTONBL': 'Vehicle Count'},
                   title="Traffic Volume Trends Over Time")
st.plotly_chart(fig_line, use_container_width=True)

# ✅ Looker Studio Integration
st.subheader("📊 Looker Studio Interactive Report")
looker_studio_url = "https://lookerstudio.google.com/embed/reporting/8b1e4dd9-a4c8-460c-8c8e-60f1dcfe2222/page/iVM4E"
st.markdown(f'<iframe src="{looker_studio_url}" width="100%" height="600px"></iframe>', unsafe_allow_html=True)

# ✅ Traffic Volume Prediction Using BigQuery ML
st.subheader("🚀 Predict Future Traffic Volume Using ML")
autosbl = st.number_input("Enter Sidebound Left Turn Count", min_value=0, value=10)
autosbt = st.number_input("Enter Sidebound Through Count", min_value=0, value=100)

if st.button("Predict Future Traffic Volume"):
    ml_query = f"""
    SELECT predicted_AUTONBL 
    FROM ML.PREDICT(
        MODEL `proj-452520.TMC.tuning_model`, 
        (SELECT {autosbl} AS AUTOSBL, {autosbt} AS AUTOSBT)
    );
    """
    try:
        ml_result = client.query(ml_query).to_dataframe()
        if not ml_result.empty and ml_result['predicted_AUTONBL'].notnull().iloc[0]:
            st.success(f"🔮 Predicted Future Traffic Volume: {ml_result['predicted_AUTONBL'].iloc[0]:.2f}")
        else:
            st.error("⚠ No prediction available. Please check input values.")
    except Exception as e:
        st.error(f"❌ Error running prediction: {e}")
