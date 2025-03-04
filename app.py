import streamlit as st
from google.cloud import bigquery
import pandas as pd
import plotly.express as px

client = bigquery.Client(project="proj-452520") 

query = """
SELECT * FROM `proj-452520.TMC.TurningMC`
LIMIT 5000
"""
df = client.query(query).to_dataframe()

df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

df_cleaned = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATE'])
df_cleaned = df_cleaned[
    (df_cleaned['LATITUDE'] > 37.0) & (df_cleaned['LATITUDE'] < 38.0) &
    (df_cleaned['LONGITUDE'] > -122.5) & (df_cleaned['LONGITUDE'] < -121.5)
]

lat_mean = df_cleaned['LATITUDE'].mean()
lon_mean = df_cleaned['LONGITUDE'].mean()

table_id = "proj-452520.TMC.CleanedTurningMC"
df_cleaned.to_gbq(destination_table=table_id, project_id="proj-452520", if_exists="replace")


st.title("Turning Movement Analysis Dashboard")
st.markdown("**Analyzing Vehicle Movements and Trends from BigQuery**")

st.sidebar.header("**Filters**")

df_cleaned = df_cleaned.dropna(subset=['DATE'])
default_start = df_cleaned['DATE'].min()
default_end = df_cleaned['DATE'].max()

date_range = st.sidebar.date_input("Select Date Range", [default_start, default_end])

filtered_df = df_cleaned[
    (df_cleaned['DATE'].dt.tz_localize(None) >= pd.to_datetime(date_range[0])) &
    (df_cleaned['DATE'].dt.tz_localize(None) <= pd.to_datetime(date_range[1]))
]

unique_intersections = df_cleaned["INTNAME"].dropna().unique().tolist()
selected_intersection = st.sidebar.selectbox("Select Intersection", ["All"] + unique_intersections)

if selected_intersection != "All":
    filtered_df = filtered_df[filtered_df["INTNAME"] == selected_intersection]

st.subheader("Top 10 Intersections with Most Vehicle Movements")

top_intersections = (
    filtered_df.groupby('INTNAME')['AUTONBL']
    .sum()
    .reset_index()
    .sort_values(by='AUTONBL', ascending=False)
    .head(10)
)

fig_bar = px.bar(
    data_frame=top_intersections,
    x="INTNAME",
    y="AUTONBL",
    labels={"INTNAME": "Intersection Name", "AUTONBL": "Total Vehicle Count"},
    title="Top 10 Intersections with Most Vehicle Movements",
)

st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart")

st.subheader("Trend of Vehicle Movements Over Time")

df_time_series = filtered_df.groupby('DATE')['AUTONBL'].sum().reset_index()

fig_line = px.line(
    df_time_series, x='DATE', y='AUTONBL',
    labels={'AUTONBL': 'Total Vehicle Count'},
    title="Trend of Vehicle Movements Over Time",
)

st.plotly_chart(fig_line, use_container_width=True, key="line_chart")

st.subheader("Traffic Flow and Geolocation")

fig_map = px.scatter_mapbox(
    filtered_df,
    lat="LATITUDE", lon="LONGITUDE",
    hover_name="INTNAME",
    color_discrete_sequence=["red"],
    zoom=10
)
fig_map.update_layout(mapbox_style="open-street-map")

st.plotly_chart(fig_map, use_container_width=True, key="map_chart")


st.subheader("**Looker Studio Interactive Report**")

looker_studio_url = "https://lookerstudio.google.com/embed/reporting/8b1e4dd9-a4c8-460c-8c8e-60f1dcfe2222/page/iVM4E"

st.markdown(
    f'<iframe src="{looker_studio_url}" width="100%" height="600px"></iframe>',
    unsafe_allow_html=True,
)

st.subheader("Traffic Movement Data Preview")
st.write(filtered_df.head())

st.write("**Forecast Vehicle Movement using BigQuery ML**")

autosbl = st.number_input("Enter Sidebound Left Turn Count", min_value=0, value=10)
autosbt = st.number_input("Enter Sidebound Through Count", min_value=0, value=100)

if st.button("Predict Future Vehicle Movement"):
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
            st.success(f"**Predicted Future Vehicle Movement: {ml_result['predicted_AUTONBL'].iloc[0]:.2f}**")
        else:
            st.error("No prediction available. Check input values!")

    except Exception as e:
        st.error(f"Error running prediction: {e}")
