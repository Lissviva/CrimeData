import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

@st.cache_data
def load_and_merge_data():
    df_main = pd.read_csv("crime_data_cleaned_with_correct_location.csv")
    df_cluster = pd.read_csv("pivot_for_clustering.csv")
    df_pred = pd.read_csv("crime_prediction_with_top_crime_type.csv")

    df_main.columns = df_main.columns.str.strip()
    df_cluster.columns = df_cluster.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()

    df_cluster_unique = df_cluster.drop_duplicates(subset='LSOA name')
    df_merged = pd.merge(df_main, df_cluster_unique, on='LSOA name', how='left')

    df_pred_unique = df_pred.drop_duplicates(subset='LSOA name')
    columns_to_add = [col for col in df_pred_unique.columns if col not in df_merged.columns or col == 'LSOA name']
    df_final = pd.merge(df_merged, df_pred_unique[columns_to_add], on='LSOA name', how='left')

    return df_final

df = load_and_merge_data()
df['Month'] = pd.to_datetime(df['Month'], errors='coerce')
df['Year'] = df['Month'].dt.year
df['Month_num'] = df['Month'].dt.month
df['Month_name'] = df['Month'].dt.strftime('%B')

pivot = pd.read_csv('pivot_for_clustering.csv')
cluster_labels = {
    0: 'Low Crime Areas',
    1: 'Mixed Crime (Moderate)',
    2: 'Residential Zones – Low Violence',
    3: 'High Crime – Shoplifting Focused'
}
pivot['Cluster Label'] = pivot['Cluster'].map(cluster_labels)
df = df.merge(pivot[['LSOA name', 'Cluster', 'Cluster Label']], on='LSOA name', how='left')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Crime Locations", "Most Common Crime", "Clustering", "Predictive Model"])

with tab1:
    st.title("Ipswich Crime (2022–2024) Dashboard")
    st.markdown("Overview of Crime in Ipswich")

    kpi1 = len(df)
    kpi2 = df['Crime type'].value_counts().idxmax()
    kpi3 = df['Location'].value_counts().idxmax()

    col1, spacer, col2, col3 = st.columns([2.5, 0.1, 4, 2.5])

    with col1:
        st.metric("Total Crimes", kpi1)
    with col2:
        st.metric("Most Common Crime", kpi2)
    with col3:
        st.metric("Top Crime Location", kpi3)

    st.markdown("---")

    col1, spacer1, col2 = st.columns([5, 0.5, 5])

    with col1:
        monthly = df.groupby(['Year', 'Month_num']).size().reset_index(name='crime_count')
        fig1 = px.line(monthly, x='Month_num', y='crime_count', color='Year', markers=True, title='Monthly Crime Trend in Ipswich')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        crime_counts = df['Crime type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        fig2 = px.pie(crime_counts.head(13), names='Crime Type', values='Count', hole=0.5, title='Top Crime Types Distribution')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

with tab2:
    st.header("Crime Locations and Resolution Outcomes")
    col3, spacer2, col4 = st.columns([1, 0.05, 1])

    with col3:
        top_streets = df['Location'].value_counts().head(10).reset_index()
        top_streets.columns = ['Location', 'Count']
        fig3 = px.bar(top_streets, x='Count', y='Location', orientation='h', title='Top 10 Streets with Most Crimes', color='Count', color_continuous_scale='Blues')
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        summary = df.groupby('Crime type')['Last outcome category'].agg(lambda x: x.value_counts().idxmax()).reset_index()
        summary['Count'] = summary.apply(lambda row: ((df['Crime type'] == row['Crime type']) & (df['Last outcome category'] == row['Last outcome category'])).sum(), axis=1)
        summary['Last outcome category'] = summary['Last outcome category'].replace({'Investigation complete; no suspect identified': 'Investigation complete;<br>no suspect identified'})
        fig4 = px.bar(summary, x='Count', y='Crime type', color='Last outcome category', orientation='h', title='Most Common Outcome per Crime Type')
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

with tab3:
    st.header("Key Insights on Sexual Offences: Streets and Seasonality")
    col5, spacer2, col6 = st.columns([1, 0.05, 1])

    with col5:
        sexual_df = df[df['Crime type'] == 'Violence and sexual offences']
        top_sexual = sexual_df['Location'].value_counts().head(10).reset_index()
        top_sexual.columns = ['Location', 'Count']
        fig7 = px.bar(top_sexual, x='Count', y='Location', orientation='h', color='Count', title='Top 10 Streets with Sexual Offences', color_continuous_scale='Reds')
        fig7.update_layout(xaxis_title='Number of Sexual Offences', yaxis_title='Street', title_font_size=18)
        st.plotly_chart(fig7, use_container_width=True)

    with col6:
        sexual_offences = df[df['Crime type'].str.strip() == 'Violence and sexual offences'].copy()
        sexual_offences['Month'] = pd.to_datetime(sexual_offences['Month'])
        monthly_sexual_crimes = sexual_offences.groupby(sexual_offences['Month'].dt.to_period('M')).size().reset_index(name='Count')
        monthly_sexual_crimes['Month'] = monthly_sexual_crimes['Month'].dt.to_timestamp()
        monthly_sexual_crimes['Year'] = monthly_sexual_crimes['Month'].dt.year
        monthly_sexual_crimes['Month_name'] = monthly_sexual_crimes['Month'].dt.strftime('%B')

        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        from pandas.api.types import CategoricalDtype
        monthly_sexual_crimes['Month_name'] = monthly_sexual_crimes['Month_name'].astype(CategoricalDtype(categories=month_order, ordered=True))

        fig8 = px.line(monthly_sexual_crimes.sort_values(by=['Year', 'Month_name']), x='Month_name', y='Count', color='Year', markers=True, title='Monthly Trend of Sexual Offences by Year', labels={'Month_name': 'Month', 'Count': 'Number of Sexual Offences'}, template='simple_white')
        fig8.update_traces(line=dict(width=2))
        fig8.update_layout(title_font_size=20)
        fig8.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")

# Puedes seguir igual para el tab4 y tab5
# Ya corregí la indentación para evitar más errores

        st.markdown("---")
