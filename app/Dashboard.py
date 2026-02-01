"""
Streamlit Application for Compensation vs Performance Analysis (Question 4).

FIX: Changed the "Underpaid Cutoff" slider to operate on a percentage scale (5-50%)
for clarity, resolving user confusion about the 0-1 range and formatting issues.

This version is stable, using caching to prevent page reloads on slider/select changes.
It is also compliant with modern Streamlit standards by using width='stretch'.

To run:
1. Ensure 'Cogentix_Case.xlsx' is in the same directory.
2. Install dependencies (streamlit, pandas, numpy, plotly, scikit-learn, openpyxl).
3. Run: streamlit run app.py
"""

import os
import math 
import pandas as pd
import numpy as np 
import streamlit as st

# plotting & dashboard
import plotly.express as px
import plotly.graph_objects as go

# modeling
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# -----------------------------
# Config / Aesthetics
# -----------------------------
st.set_page_config(layout="wide", page_title="Compensation vs Performance Analysis")

PRIMARY_COLOR = '#005F99' 
ACCENT_COLOR = '#00B8AA' 
SEMANTIC_GREEN = '#28a745'
SEMANTIC_RED = '#dc3545'

DATA_PATH = "../data/Cogentix_Case.xlsx" 
DEFAULT_QUANTILE = 0.20 # Used to set the default slider position (20%)

# -----------------------------
# Data Loading and Modeling (CACHED)
# -----------------------------

@st.cache_data(show_spinner="Loading data and cleaning features...")
def load_data():
    """Loads and performs initial cleaning on the data."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}. Please ensure the Excel file is in the same directory.")
        return pd.DataFrame(), 0
    
    try:
        df = pd.read_excel(DATA_PATH, sheet_name=0, engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return pd.DataFrame(), 0

    # --- Robust Column Handling & Cleaning ---
    df['Location_Clean'] = df.get('Location', pd.Series(['Unknown']*len(df))).astype(str)
    df['Department'] = df.get('Department', pd.Series(['Unknown']*len(df))).fillna('Unknown').astype(str)
    df['Role'] = df.get('Role', pd.Series(['Unknown']*len(df))).fillna('Unknown').astype(str)
    df['Name'] = df.get('Name', pd.Series(['']*len(df))).fillna('').astype(str)
    df['EmployeeID'] = df.get('EmployeeID', df.index)
    
    df['AnnualSalaryINR'] = df.get('AnnualSalaryINR', pd.Series([0.0]*len(df))).astype(float).fillna(0.0)
    df['PerformanceRating'] = df.get('PerformanceRating', pd.Series([3.0]*len(df))).astype(float).fillna(3.0)
    df['BonusPercent'] = df.get('BonusPercent', pd.Series([0.0]*len(df))).fillna(0.0) 
    df['YearsAtCompany'] = df.get('YearsAtCompany', pd.Series([0]*len(df))).fillna(0)
    
    df['AnnualSalary_k'] = df['AnnualSalaryINR'] / 1000.0
    
    max_perf = df['PerformanceRating'].max()
    dynamic_perf_cut = int(max_perf - 1) if max_perf >= 4 else int(max_perf) 
    
    return df, dynamic_perf_cut

@st.cache_resource(show_spinner="Training Compensation Residual Model...")
def train_residual_model(df):
    """Trains the Linear Regression model and calculates salary residuals."""
    if df.empty:
        return df
        
    # --- Residual Model Calculation ---
    model_df = pd.DataFrame({
        'PerformanceRating': df['PerformanceRating'],
        'YearsAtCompany': df['YearsAtCompany'] 
    })
    model_df['Role'] = df['Role'].astype(str)
    model_df['Location_Clean'] = df['Location_Clean'].astype(str)

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    cat_feats = ['Role', 'Location_Clean']
    cat_arr = ohe.fit_transform(model_df[cat_feats])

    X_num = model_df[['PerformanceRating', 'YearsAtCompany']].to_numpy()
    X = np.hstack([X_num, cat_arr])
    # Log transformation on salary for better linear modeling
    y = np.log(df['AnnualSalaryINR'].astype(float) + 1.0) 

    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    
    # Residual: The difference between actual log-salary and predicted log-salary
    df['salary_resid'] = y - y_pred 
    
    return df

# Load and process data (runs only once)
df_base_raw, INITIAL_PERF_CUT = load_data()
df_base = train_residual_model(df_base_raw.copy())


# -----------------------------
# Helper Functions
# -----------------------------

def compute_underpaid(df_in, quantile, perf_cut):
    """Identifies employees who are high performers AND have low salary residuals."""
    if len(df_in) == 0:
        return pd.DataFrame(columns=df_in.columns), float('nan')
    
    thr = df_in['salary_resid'].quantile(quantile) 
    
    up = df_in[
        (df_in['PerformanceRating'] >= perf_cut) & 
        (df_in['salary_resid'] <= thr)
    ].copy()
    
    up = up.sort_values('salary_resid').reset_index(drop=True)
    return up, thr

def update_fig_layout(fig, height=None):
    """
    Applies a clean, consistent layout to all figures.
    Ensures dark hover text for visibility.
    """
    fig.update_layout(
        template='plotly_white', 
        title_font_color=PRIMARY_COLOR,
        font_family="Arial, sans-serif",
        margin={'l': 40, 'r': 10, 't': 40, 'b': 40},
        # Fixed font_color to dark grey for better readability
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial", font_color='#333333'), 
        height=height
    )
    return fig

# -----------------------------
# Figure Builders 
# -----------------------------

def fig_salary_vs_perf(dff):
    fig = px.scatter(
        dff, x='AnnualSalary_k', y='PerformanceRating',
        color='Department', size='BonusPercent',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=['EmployeeID','Name','Role','AnnualSalaryINR'],
        title='💸 Salary (k INR) vs Performance Rating'
    )
    fig.update_layout(xaxis_title='Annual Salary (k INR)', yaxis_title='Performance Rating', 
                      legend_title='Department')
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.2, color='Gray')))
    return update_fig_layout(fig, height=500)

def fig_dept_summary(dff):
    if dff.empty:
        fig = go.Figure().update_layout(title="🏢 Department Summary (No data)")
        return update_fig_layout(fig, height=500)

    dept = dff.groupby('Department').agg(
        avg_salary=('AnnualSalaryINR','mean'),
        avg_perf=('PerformanceRating','mean'),
        headcount=('EmployeeID','count')
    ).reset_index()
    fig = px.scatter(dept, x='avg_salary', y='avg_perf', size='headcount', text='Department',
                     color='avg_perf',
                     color_continuous_scale=px.colors.sequential.Plotly3,
                     hover_data=['Department','headcount'], 
                     title='🏢 Department: Avg Salary vs Avg Performance')
    fig.update_traces(textposition='top center')
    fig.update_layout(xaxis_title='Avg Annual Salary (INR)', yaxis_title='Avg Performance Rating')
    return update_fig_layout(fig, height=500)

def fig_salary_violin(dff):
    fig = px.violin(
        dff, y='AnnualSalaryINR', x='PerformanceRating', 
        color='PerformanceRating', 
        color_discrete_sequence=px.colors.sequential.Teal,
        box=True, points='outliers',
        hover_data=['EmployeeID','Role'],
        title='🎻 Salary distribution by Performance Rating'
    )
    fig.update_yaxes(tickformat=',', title='Annual Salary (INR)')
    return update_fig_layout(fig, height=400)

def fig_residuals_violin(dff, thr_val):
    fig = px.violin(dff, x='PerformanceRating', y='salary_resid', 
                     color='PerformanceRating', 
                     color_discrete_sequence=px.colors.sequential.Teal,
                     box=True, points='outliers',
                     hover_data=['EmployeeID','Name','Role'], 
                     title='📉 Salary Residuals by Performance')
    
    if not math.isnan(thr_val):
        # Display the quantile as a percentage in the annotation
        quantile_percent = int(100 * DEFAULT_QUANTILE) # This is 20 for default
        fig.add_hline(y=thr_val, line_dash='dash', line_color=SEMANTIC_RED, 
                      annotation_text=f'Underpaid Cutoff ({quantile_percent}% Resid)', 
                      annotation_position='top left',
                      annotation_font_color=SEMANTIC_RED)
    
    fig.update_layout(yaxis_title='Log-salary residual (Actual - Predicted)')
    return update_fig_layout(fig, height=400)

def fig_resid_hist(dff, quantile_val):
    fig = px.histogram(dff, x='salary_resid', nbins=60, 
                       color_discrete_sequence=[PRIMARY_COLOR],
                       title='📊 Salary Residuals Distribution')
    
    if not dff.empty and quantile_val is not None:
        try:
            cutoff = dff['salary_resid'].quantile(quantile_val)
            # Display the quantile as a percentage in the annotation
            quantile_percent = int(100 * quantile_val)
            fig.add_vline(x=cutoff, line_dash='dash', line_color=SEMANTIC_RED,
                          annotation_text=f'{quantile_percent}% Cutoff', annotation_position='top left',
                          annotation_font_color=SEMANTIC_RED)
        except:
            pass
    
    return update_fig_layout(fig, height=350)


# -----------------------------
# Streamlit Layout
# -----------------------------

st.title("Compensation vs Performance Analysis 📊")

if df_base.empty:
    st.stop()

# --- SIDEBAR FILTERS ---
all_departments = sorted(df_base['Department'].dropna().unique().tolist())
perf_min_init = int(df_base['PerformanceRating'].min())

with st.sidebar:
    st.header("Filter & Analysis Controls")

    dept_values = st.multiselect(
        "Department (Select one or more)",
        options=all_departments,
        default=None,
        key='dept_filter'
    )

    if dept_values:
        roles_in_selected = sorted(df_base[df_base['Department'].isin(dept_values)]['Role'].dropna().unique().tolist())
        role_values = st.multiselect(
            "Role (Updates after department selection)",
            options=roles_in_selected,
            default=None,
            key='role_filter'
        )
    else:
        role_values = None
        st.multiselect(
            "Role (Select Department first)",
            options=[],
            disabled=True
        )

    st.markdown("---")
    st.subheader("Underpaid Flagging Criteria")

    # 3. Cutoff Slider
    # FIX: Slider now operates on the percentage scale (5-50) for clarity.
    cutoff_q_percent = st.slider(
        "Underpaid Cutoff (Bottom % of Residuals)",
        min_value=5, max_value=50, step=5, 
        value=int(DEFAULT_QUANTILE * 100), # Use 20 for default
        format='%d%%'
    )
    # Convert the percentage value back to the quantile (0.05 - 0.50) for calculation
    cutoff_q = cutoff_q_percent / 100.0 

    # 4. Performance Min Slider
    perf_min = st.slider(
        "Min Performance Rating Filter",
        min_value=perf_min_init, 
        max_value=int(df_base['PerformanceRating'].max()),
        step=1, 
        value=INITIAL_PERF_CUT
    )

# --- APPLY FILTERS ---
dff = df_base.copy()

if dept_values:
    dff = dff[dff['Department'].isin(dept_values)]

if role_values:
    dff = dff[dff['Role'].isin(role_values)]

perf_cut = perf_min 
dff_filtered_perf = dff[dff['PerformanceRating'] >= perf_cut].copy()

# Note: cutoff_q is now a clean decimal value (e.g., 0.20)
underpaid_df, residual_threshold = compute_underpaid(dff_filtered_perf, quantile=cutoff_q, perf_cut=perf_cut)

# --- KPI ROW ---
col1, col2, col3 = st.columns(3)

avg_sal = dff_filtered_perf['AnnualSalaryINR'].mean() if len(dff_filtered_perf)>0 else 0
avg_perf = dff_filtered_perf['PerformanceRating'].mean() if len(dff_filtered_perf)>0 else 0
underpaid_n = len(underpaid_df)
total = len(dff_filtered_perf)

with col1:
    st.metric(label="Avg Salary (INR)", value=f"₹{int(avg_sal):,}")
with col2:
    st.metric(label="Avg Performance", value=f"⭐ {avg_perf:.2f}")
with col3:
    st.metric(label="Flagged Underpaid High Perf", value=f"🚨 {underpaid_n:,} / {total:,}", delta_color="inverse")

st.markdown("---")

# --- GRAPH ROW 1 (Scatter vs Dept Summary) ---
col_scatter, col_dept = st.columns([7, 5])
with col_scatter:
    st.plotly_chart(fig_salary_vs_perf(dff_filtered_perf), width='stretch')
with col_dept:
    st.plotly_chart(fig_dept_summary(dff_filtered_perf), width='stretch')

st.markdown("---")

# --- GRAPH ROW 2 (Violin Plots) ---
col_violin_sal, col_violin_resid = st.columns(2)
with col_violin_sal:
    st.plotly_chart(fig_salary_violin(dff_filtered_perf), width='stretch')
with col_violin_resid:
    # Pass the calculated quantile back for correct annotation
    st.plotly_chart(fig_residuals_violin(dff_filtered_perf, residual_threshold), width='stretch')

st.markdown("---")

# --- TABLE AND HISTOGRAM ---
col_hist, col_table = st.columns(2)

with col_hist:
    # Pass the calculated quantile back for correct annotation
    st.plotly_chart(fig_resid_hist(dff_filtered_perf, cutoff_q), width='stretch')

with col_table:
    st.subheader("Underpaid High Performers 🧐")
    
    cols_for_table = ['EmployeeID','Name','Department','Role','PerformanceRating','AnnualSalaryINR','salary_resid']
    underpaid_display = underpaid_df[[c for c in cols_for_table if c in underpaid_df.columns]].copy()
    
    underpaid_display['AnnualSalaryINR'] = underpaid_display['AnnualSalaryINR'].apply(lambda x: f"₹{int(x):,}")
    underpaid_display['salary_resid'] = underpaid_display['salary_resid'].round(2)
    
    st.dataframe(underpaid_display, use_container_width=True, hide_index=True) 
    
    csv_export = underpaid_df[cols_for_table].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Underpaid CSV List",
        data=csv_export,
        file_name='underpaid_high_performers.csv',
        mime='text/csv',
        key='download_button'
    )