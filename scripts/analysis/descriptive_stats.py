import pandas as pd
import streamlit as st

@st.cache_data
def get_descriptive_stats(df, variable):
    stats = df[variable].describe().to_frame("Statistiques")
    return stats
