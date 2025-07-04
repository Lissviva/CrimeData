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
def load_data():
    df = pd.read_csv("crime_data_cleaned_with_correct_location.csv", parse_dates=['Month'])
    return df

@st.cache_data
def load_prediction_data():
    df = pd.read_csv("crime_data_for_prediction.csv")
    df['District'] = df['District'].str.strip()
    return df


df = load_data()
df.columns = df.columns.str.strip()
df['Year'] = df['Month'].dt.year
df['Month_num'] = df['Month'].dt.month
df['Month_name'] = df['Month'].dt.strftime('%B')

# 3. Load and merge clusters
pivot = pd.read_csv('pivot_for_clustering.csv')
cluster_labels = {
    0: 'Low Crime Areas',
    1: 'Mixed Crime (Moderate)',
    2: 'Residential Zones – Low Violence',
    3: 'High Crime – Shoplifting Focused'
}
pivot['Cluster Label'] = pivot['Cluster'].map(cluster_labels)
df = df.merge(pivot[['LSOA name', 'Cluster', 'Cluster Label']], on='LSOA name', how='left')

# 4. Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Crime Locations", "Most Common Crime", "Clustering", "Predictive Model"])

# ------------------------- TAB 1: EDA Overview -------------------------
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
        fig1 = px.line(
            monthly, x='Month_num', y='crime_count',
            color='Year', markers=True, title='Monthly Crime Trend in Ipswich'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        crime_counts = df['Crime type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        fig2 = px.pie(
            crime_counts.head(13),
            names='Crime Type', values='Count',
            hole=0.5, title='Top Crime Types Distribution'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

# ------------------------- TAB 2: Crime Locations -------------------------
with tab2:
    st.header("Crime Locations and Resolution Outcomes")
    col3, spacer2, col4 = st.columns([1, 0.05, 1])

    with col3:
        top_streets = df['Location'].value_counts().head(10).reset_index()
        top_streets.columns = ['Location', 'Count']
        fig3 = px.bar(
            top_streets, x='Count', y='Location',
            orientation='h', title='Top 10 Streets with Most Crimes',
            color='Count', color_continuous_scale='Blues'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        summary = (
            df.groupby('Crime type')['Last outcome category']
            .agg(lambda x: x.value_counts().idxmax()).reset_index()
        )
        summary['Count'] = summary.apply(
            lambda row: ((df['Crime type'] == row['Crime type']) & 
                         (df['Last outcome category'] == row['Last outcome category'])).sum(), axis=1
        )
        summary['Last outcome category'] = summary['Last outcome category'].replace({
            'Investigation complete; no suspect identified': 'Investigation complete;<br>no suspect identified'
        })
        fig4 = px.bar(
            summary, x='Count', y='Crime type',
            color='Last outcome category', orientation='h',
            title='Most Common Outcome per Crime Type'
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

# ------------------------- TAB 3: Top Crime Focus -------------------------
with tab3:
    st.header("Key Insights on Sexual Offences: Streets and Seasonality")
    col5, spacer2, col6 = st.columns([1, 0.05, 1])
    
with col5:
    # Filter only sexual offences
    sexual_df = df[df['Crime type'] == 'Violence and sexual offences']

    # Top 10 streets with most sexual offences
    top_sexual = sexual_df['Location'].value_counts().head(10).reset_index()
    top_sexual.columns = ['Location', 'Count']

    # Create chart
    fig7 = px.bar(
        top_sexual,
        x='Count', y='Location',
        orientation='h',
        color='Count',
        title='Top 10 Streets with Sexual Offences',
        color_continuous_scale='Reds'
    )
    fig7.update_layout(
        xaxis_title='Number of Sexual Offences',
        yaxis_title='Street',
        title_font_size=18
    )

    st.plotly_chart(fig7, use_container_width=True)
    
with col6:
    # Filter only sexual offences
    sexual_offences = df[df['Crime type'].str.strip() == 'Violence and sexual offences'].copy()

    # Ensure 'Month' column is datetime
    sexual_offences['Month'] = pd.to_datetime(sexual_offences['Month'])

    # Group by month
    monthly_sexual_crimes = (
        sexual_offences
        .groupby(sexual_offences['Month'].dt.to_period('M'))
        .size()
        .reset_index(name='Count')
    )
    monthly_sexual_crimes['Month'] = monthly_sexual_crimes['Month'].dt.to_timestamp()

    # Extract year and month name
    monthly_sexual_crimes['Year'] = monthly_sexual_crimes['Month'].dt.year
    monthly_sexual_crimes['Month_name'] = monthly_sexual_crimes['Month'].dt.strftime('%B')

    # Order months
    from pandas.api.types import CategoricalDtype
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_sexual_crimes['Month_name'] = monthly_sexual_crimes['Month_name'].astype(
        CategoricalDtype(categories=month_order, ordered=True)
    )

    # Create line chart
    fig8 = px.line(
        monthly_sexual_crimes.sort_values(by=['Year', 'Month_name']),
        x='Month_name',
        y='Count',
        color='Year',
        markers=True,
        title='Monthly Trend of Sexual Offences by Year',
        labels={'Month_name': 'Month', 'Count': 'Number of Sexual Offences'},
        template='simple_white'
    )

    fig8.update_traces(line=dict(width=2))
    fig8.update_layout(title_font_size=20)
    fig8.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    st.plotly_chart(fig8, use_container_width=True) 
    
    st.markdown("---")

# ------------------------- TAB 4: Clustering -------------------------
with tab4:
    st.header("Socioeconomic Indicators and Cluster Analysis")

    col7, spacer3, col8 = st.columns([1, 0.05, 1])

    # --- Left column: Line chart ---
    with col7:
        st.subheader("Annual Trend: Total Crimes vs Economic Inactivity Rate")

        inactivity_trend = df.groupby('Year').agg({
            'Crime ID': 'count',
            'economic_inactivity_rate': 'mean'
        }).rename(columns={'Crime ID': 'Total Crimes'}).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=inactivity_trend['Year'],
            y=inactivity_trend['Total Crimes'],
            name='Total Crimes',
            mode='lines+markers',
            line=dict(color='red', width=3),
            yaxis='y1'
        ))

        fig.add_trace(go.Scatter(
            x=inactivity_trend['Year'],
            y=inactivity_trend['economic_inactivity_rate'],
            name='Economic Inactivity Rate (%)',
            mode='lines+markers',
            line=dict(color='blue', width=3),
            yaxis='y2'
        ))

        fig.update_layout(
            title="Annual Trend: Total Crimes vs Economic Inactivity Rate",
            xaxis=dict(title='Year'),
            yaxis=dict(title='Total Crimes', titlefont=dict(color='red'), tickfont=dict(color='red')),
            yaxis2=dict(title='Economic Inactivity Rate (%)', titlefont=dict(color='blue'),
                        tickfont=dict(color='blue'), overlaying='y', side='right'),
            legend=dict(orientation='h', yanchor='top', y=-0.25, xanchor='center', x=0.5),
            margin=dict(t=50, l=60, r=60, b=100)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- Right column: Cluster map ---
    with col8:
        st.subheader("Clustered Crime Heatmap by Area")
        st.markdown("""
        <div style='font-size: 15px; line-height: 1.6'>
        <span style="color:purple">⬤</span> <b>Mixed Crime (Moderate)</b>: Areas with varied but moderate levels of crime.<br>
        <span style="color:orange">⬤</span> <b>Low Violence</b>: Mainly quiet neighborhoods with occasional incidents.<br>
        <span style="color:green">⬤</span> <b>Low Crime Areas</b>: Generally safe areas with minimal reported crime.<br>
        <span style="color:red">⬤</span> <b>High Crime – Shoplifting Focused</b>: Busy commercial areas with frequent theft reports.
        </div>
        """, unsafe_allow_html=True)

        clusters = df[['LSOA name', 'Latitude', 'Longitude', 'Cluster Label']].drop_duplicates()
        cluster_map = folium.Map(location=[52.056, 1.148], zoom_start=13)

        color_map = {
            'Mixed Crime (Moderate)': 'purple',
            'Residential Zones – Low Violence': 'orange',
            'Low Crime Areas': 'green',
            'High Crime – Shoplifting Focused': 'red'
        }

        for _, row in clusters.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=color_map.get(row['Cluster Label'], 'gray'),
                fill=True,
                fill_opacity=0.7,
                popup=row['Cluster Label']
            ).add_to(cluster_map)

        folium_static(cluster_map, width=600)
    
# ------------------------- TAB 5: Predictive Model -------------------------
with tab5:
    st.header("Predictive Insights")

    col9, spacer4, col10 = st.columns([5, 0.5, 5])

    with col9:
        st.subheader("Average Predicted Crime Risk in Ipswich")

        @st.cache_data
        def load_prediction_top_crime():
            df = pd.read_csv('crime_prediction_with_top_crime_type.csv')
            df['District'] = df['District'].str.strip()
            return df

        proba_df = load_prediction_top_crime()

        exclude_cols = ['Top Crime Type', 'District', 'Latitude', 'Longitude', 'Year', 'Month']
        crime_type_cols = [col for col in proba_df.columns if col not in exclude_cols]

        ipswich_avg = proba_df[proba_df['District'] == 'Ipswich'][crime_type_cols].mean().sort_values(ascending=False)
        ipswich_avg_df = ipswich_avg.reset_index()
        ipswich_avg_df.columns = ['Crime type', 'Predicted Probability']

        plt.figure(figsize=(10, 6))
        sns.barplot(data=ipswich_avg_df, x='Predicted Probability', y='Crime type', palette='viridis')
        plt.title("Average Predicted Crime Risk in Ipswich", fontsize=14)
        plt.xlabel("Predicted Probability")
        plt.ylabel("Crime Type")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

    with col10:
        st.subheader("Top Predicted Crime Type per Area")
        st.markdown("Click on a circle to see the specific predicted crime types in that zone.")

        df_pred = proba_df.dropna(subset=['Latitude', 'Longitude', 'Top Crime Type'])

        crime_types = sorted(df_pred['Top Crime Type'].unique())
        colors = px.colors.qualitative.Set3
        crime_colors = {crime: colors[i % len(colors)] for i, crime in enumerate(crime_types)}

        m = folium.Map(location=[52.0567, 1.1482], zoom_start=13)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df_pred.iterrows():
            crime_type = row['Top Crime Type']
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5,
                color=crime_colors.get(crime_type, 'gray'),
                fill=True,
                fill_opacity=0.6,
                popup=folium.Popup(f"<b>Predicted:</b> {crime_type}", max_width=200)
            ).add_to(marker_cluster)

        folium_static(m, width=600)

        st.markdown("---")