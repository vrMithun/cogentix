# Dashboard_Q4_Compensation_Analysis_Final_Guaranteed_Alignment.py
"""
Dash app for Question 4 (Compensation vs Performance) with dependent dropdowns 
and enhanced professional styling.

Status: Verified and robustified. KPI alignment guaranteed via strict Bootstrap 
col-md-4 and d-flex usage.
"""

import os
import math 
import pandas as pd
import numpy as np 

# plotting & dashboard
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table

# modeling
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# -----------------------------
# Config / Aesthetics
# -----------------------------

# Define a professional color palette
PRIMARY_COLOR = '#005F99' # Dark Corporate Blue
ACCENT_COLOR = '#00B8AA' # Teal/Green for contrast
SEMANTIC_GREEN = '#28a745'
SEMANTIC_RED = '#dc3545'

# NOTE: Update this path if your file location changes.
# IMPORTANT: Replace this placeholder path with your actual Excel file path.
DATA_PATH = r"\Cogentix_Case.xlsx"
ASSETS_DIR = "assets"
UNDERPAID_CSV = os.path.join(ASSETS_DIR, "underpaid_high_performers.csv")

# Create assets directory if it doesn't exist
os.makedirs(ASSETS_DIR, exist_ok=True)

# -----------------------------
# Data Loading and Preprocessing
# -----------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Update DATA_PATH accordingly.")

try:
    df = pd.read_excel(DATA_PATH, sheet_name=0, engine='openpyxl')
    print("Loaded file. Columns:", df.columns.tolist())
except Exception as e:
    raise RuntimeError(f"Error loading Excel file: {e}")

# --- Core Required Columns Checks ---
if 'AnnualSalaryINR' not in df.columns:
    raise RuntimeError("AnnualSalaryINR missing - required.")
if 'PerformanceRating' not in df.columns:
    raise RuntimeError("PerformanceRating missing - required.")

# --- Robust Column Handling ---
# Use .get() with defaults for optional/standardized columns
df['Location_Clean'] = df.get('Location', pd.Series(['Unknown']*len(df))).astype(str)
df['Department'] = df.get('Department', pd.Series(['Unknown']*len(df))).fillna('Unknown').astype(str)
df['Role'] = df.get('Role', pd.Series(['Unknown']*len(df))).fillna('Unknown').astype(str)
df['Name'] = df.get('Name', pd.Series(['']*len(df))).fillna('').astype(str)
df['ManagerID'] = df.get('ManagerID', pd.Series([None]*len(df)))
df['EmployeeID'] = df.get('EmployeeID', df.index) # Default EmployeeID if missing
df['EngagementScore'] = df.get('EngagementScore', pd.Series([0.0]*len(df))).fillna(0.0)

# Salary/Performance setup
df['AnnualSalary_k'] = df['AnnualSalaryINR'] / 1000.0
df['PerformanceRating'] = df['PerformanceRating'].astype(float)
df['BonusPercent'] = df.get('BonusPercent', pd.Series([0.0]*len(df))).fillna(0.0) 
df['YearsAtCompany'] = df.get('YearsAtCompany', pd.Series([0]*len(df))).fillna(0)

# Residual Model Calculation
model_df = pd.DataFrame({
    'PerformanceRating': df['PerformanceRating'].fillna(df['PerformanceRating'].median()),
    'YearsAtCompany': df['YearsAtCompany'] 
})
model_df['Role'] = df['Role'].astype(str)
model_df['Location_Clean'] = df['Location_Clean'].astype(str)

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
cat_feats = ['Role', 'Location_Clean']
cat_arr = ohe.fit_transform(model_df[cat_feats])

X_num = model_df[['PerformanceRating', 'YearsAtCompany']].to_numpy()
X = np.hstack([X_num, cat_arr])
y = np.log(df['AnnualSalaryINR'].astype(float) + 1.0) 

lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
df['salary_pred_log'] = y_pred
df['salary_log'] = y
df['salary_resid'] = df['salary_log'] - df['salary_pred_log'] 

# Underpaid-High-Performer Helper
DEFAULT_QUANTILE = 0.20
max_perf = df['PerformanceRating'].max()
DEFAULT_PERF_CUT = int(max_perf - 1) if max_perf >= 4 else int(max_perf) 

def compute_underpaid(df_in, quantile=DEFAULT_QUANTILE, perf_cut=DEFAULT_PERF_CUT):
    if len(df_in) == 0:
        return pd.DataFrame(columns=df_in.columns), float('nan')
    
    thr = df_in['salary_resid'].quantile(quantile) 
    
    up = df_in[
        (df_in['PerformanceRating'] >= perf_cut) & 
        (df_in['salary_resid'] <= thr)
    ].copy()
    
    up = up.sort_values('salary_resid').reset_index(drop=True)
    return up, thr

# -----------------------------
# Figure Builders
# -----------------------------
def update_fig_layout(fig):
    """Applies a clean, consistent layout to all figures."""
    fig.update_layout(
        template='plotly_white', 
        title_font_color=PRIMARY_COLOR,
        font_family="Arial, sans-serif",
        margin={'l': 40, 'r': 10, 't': 40, 'b': 40},
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )
    return fig

def fig_salary_vs_perf(dff):
    fig = px.scatter(
        dff, x='AnnualSalary_k', y='PerformanceRating',
        color='Department', size='BonusPercent',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=['EmployeeID','Name','Role','ManagerID','EngagementScore','YearsAtCompany','AnnualSalaryINR'],
        title='💸 Salary (k INR) vs Performance Rating'
    )
    fig.update_layout(xaxis_title='Annual Salary (k INR)', yaxis_title='Performance Rating', 
                      legend_title='Department', height=500)
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.2, color='Gray')))
    return update_fig_layout(fig)

def fig_dept_summary(dff):
    if dff.empty:
        return update_fig_layout(go.Figure().update_layout(title="🏢 Department Summary (No data)"))

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
    fig.update_layout(xaxis_title='Avg Annual Salary (INR)', yaxis_title='Avg Performance Rating', 
                      height=500)
    return update_fig_layout(fig)

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
    fig.update_layout(height=400)
    return update_fig_layout(fig)

def fig_residuals_violin(dff, thr_val):
    fig = px.violin(dff, x='PerformanceRating', y='salary_resid', 
                    color='PerformanceRating', 
                    color_discrete_sequence=px.colors.sequential.Teal,
                    box=True, points='outliers',
                    hover_data=['EmployeeID','Name','Role','ManagerID'], 
                    title='📉 Salary Residuals by Performance')
    
    if not math.isnan(thr_val):
        fig.add_hline(y=thr_val, line_dash='dash', line_color=SEMANTIC_RED, 
                      annotation_text=f'Underpaid Cutoff ({int(100*DEFAULT_QUANTILE)}% Resid)', 
                      annotation_position='top left',
                      annotation_font_color=SEMANTIC_RED)
    
    fig.update_layout(yaxis_title='Log-salary residual (Actual - Predicted)', height=400)
    return update_fig_layout(fig)

def fig_resid_hist(dff, quantile_val):
    fig = px.histogram(dff, x='salary_resid', nbins=60, 
                       color_discrete_sequence=[PRIMARY_COLOR],
                       title='📊 Salary Residuals Distribution')
    
    if not dff.empty and quantile_val is not None:
        fig.add_vline(x=dff['salary_resid'].quantile(quantile_val), line_dash='dash', line_color=SEMANTIC_RED,
                      annotation_text=f'{int(100*quantile_val)}% Cutoff', annotation_position='top left',
                      annotation_font_color=SEMANTIC_RED)
    
    fig.update_layout(height=350)
    return update_fig_layout(fig)

# KPI cards (FINAL, GUARANTEED ALIGNMENT)
def kpi_cards(df_in, underpaid_df):
    avg_sal = df_in['AnnualSalaryINR'].mean() if len(df_in)>0 else 0
    avg_perf = df_in['PerformanceRating'].mean() if len(df_in)>0 else 0
    underpaid_n = len(underpaid_df)
    total = len(df_in)
    
    # Formatting 
    formatted_avg_sal = f"₹{int(avg_sal):,}"
    formatted_avg_perf = f"{avg_perf:.2f}"
    formatted_underpaid = f"{underpaid_n:,} / {total:,}"

    card_style_base = {
        'padding':'15px',
        # Removed individual margins (marginLeft/Right) to rely purely on the Bootstrap grid
        'border':'1px solid #d4d4d4',
        'borderRadius':'8px','textAlign':'center', 
        'boxShadow': '2px 2px 5px rgba(0,0,0,0.1)',
        'backgroundColor': '#FFFFFF',
        'height': '100%',
        # Flex properties to center content vertically inside the card
        'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'minHeight': '120px'
    }
    
    # Parent row classes: 'row' (grid system), 'd-flex align-items-stretch' (equal height), 
    # 'mx-0' (removes default row side margins that caused wrapping issues)
    cards = html.Div([
        # KPI 1: Avg Salary (col-md-4) - use 'p-2' for inner spacing instead of custom margins
        html.Div([
            html.H4("Avg Salary (INR)", style={'color':PRIMARY_COLOR, 'fontSize':'1.1em', 'fontWeight':'normal'}), 
            html.P(formatted_avg_sal, style={'fontSize':'2.2em', 'fontWeight':'bold', 'color': PRIMARY_COLOR})
        ], style={**card_style_base}, className='col-md-4 p-2'),

        # KPI 2: Avg Performance (col-md-4)
        html.Div([
            html.H4("Avg Performance", style={'color':PRIMARY_COLOR, 'fontSize':'1.1em', 'fontWeight':'normal'}), 
            html.P(html.Span(['⭐ ', formatted_avg_perf]), style={'fontSize':'2.2em', 'fontWeight':'bold', 'color': SEMANTIC_GREEN})
        ], style={**card_style_base}, className='col-md-4 p-2'),
        
        # KPI 3: Flagged Underpaid (col-md-4)
        html.Div([
            html.H4("Flagged Underpaid High Perf", style={'color':PRIMARY_COLOR, 'fontSize':'1.1em', 'fontWeight':'normal'}), 
            html.P(html.Span(['🚨 ', formatted_underpaid]), style={'fontSize':'2.2em', 'fontWeight':'bold', 'color': SEMANTIC_RED})
        ], style={**card_style_base}, className='col-md-4 p-2')
        
    ], className='row d-flex align-items-stretch mx-0') 
    return cards

# -----------------------------
# Dash app layout
# -----------------------------
app = Dash(__name__, assets_folder=ASSETS_DIR, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])
server = app.server

all_departments = sorted(df['Department'].dropna().unique().tolist())
PERF_MIN_INIT = int(df['PerformanceRating'].min())

app.layout = html.Div([
    html.H1("Compensation vs Performance Analysis 📊", className='text-center my-4', style={'color':PRIMARY_COLOR}),
    
    # KPI Row (The 'mx-0' class is crucial for removing side margins and is applied inside the kpi_cards function's wrapper)
    html.Div(id='kpis', className='mx-0'), 
    
    html.Hr(style={'borderColor': PRIMARY_COLOR, 'borderWidth':'2px', 'margin': '20px 0'}),

    # Control Panel
    html.Div([
        # Department Filter
        html.Div([
            html.Label("Department (Select one or more)", className='font-weight-bold', style={'color':PRIMARY_COLOR}),
            dcc.Dropdown(id='dept-filter',
                         options=[{'label': d, 'value': d} for d in all_departments],
                         value=None, multi=True, placeholder="Select department(s) first", className='mb-3')
        ], className='col-md-4'),

        # Role Filter
        html.Div([
            html.Label("Role (Updates after department selection)", className='font-weight-bold', style={'color':PRIMARY_COLOR}),
            dcc.Dropdown(id='role-filter',
                         options=[], value=None, multi=True, placeholder="Select role(s) (disabled until dept selected)",
                         disabled=True, className='mb-3')
        ], className='col-md-4'),

        # Cutoff / Perf Sliders
        html.Div([
            html.Label("Underpaid Cutoff (Quantile of Residuals)", className='font-weight-bold', style={'color':PRIMARY_COLOR}),
            dcc.Slider(id='cutoff-quantile', min=0.05, max=0.5, step=0.05, value=DEFAULT_QUANTILE,
                       marks={q:f'{int(q*100)}%' for q in [0.05, 0.1, 0.2, 0.3, 0.5]},
                       tooltip={"placement": "bottom", "always_visible": True}),
            html.Br(),
            html.Label("Min Performance Rating Filter", className='font-weight-bold', style={'color':PRIMARY_COLOR}),
            dcc.Slider(id='perf-min', 
                       min=PERF_MIN_INIT, 
                       max=int(df['PerformanceRating'].max()),
                       step=1, 
                       value=DEFAULT_PERF_CUT,
                       marks={i:str(i) for i in range(PERF_MIN_INIT, int(df['PerformanceRating'].max())+1)},
                       tooltip={"placement": "bottom", "always_visible": True})
        ], className='col-md-4')
    ], className='row px-2 pb-0'),
    
    html.Hr(style={'borderColor': PRIMARY_COLOR, 'borderWidth':'2px', 'margin': '20px 0'}),

    # Graph Row 1
    html.Div([
        html.Div(dcc.Graph(id='scatter-salary-perf'), className='col-md-7'),
        html.Div(dcc.Graph(id='dept-summary'), className='col-md-5')
    ], className='row'),
    
    html.Hr(style={'borderColor': PRIMARY_COLOR, 'borderWidth':'1px'}),

    # Graph Row 2
    html.Div([
        html.Div(dcc.Graph(id='violin-salary'), className='col-md-6'),
        html.Div(dcc.Graph(id='resid-violin'), className='col-md-6')
    ], className='row'),
    
    html.Hr(style={'borderColor': PRIMARY_COLOR, 'borderWidth':'1px'}),

    # Table / Hist Row
    html.Div([
        html.Div(dcc.Graph(id='resid-hist'), className='col-md-6'),
        html.Div([
            html.H3("Underpaid High Performers 🧐", className='mb-3', style={'color':PRIMARY_COLOR}),
            html.A("Download underpaid CSV", href=f"/assets/{os.path.basename(UNDERPAID_CSV)}", target="_blank", className='btn btn-primary btn-sm mb-3'),
            dash_table.DataTable(id='underpaid-table',
                                 columns=[{"name": c, "id": c, "type": "numeric", "format": dash_table.Format.Format(precision=2, scheme=dash_table.Format.Scheme.decimal_or_exponent)} 
                                          for c in ['EmployeeID','Name','Department','Role','ManagerID','PerformanceRating','AnnualSalaryINR','salary_resid']],
                                 page_size=10, sort_action='native', filter_action='native', 
                                 style_table={'overflowX':'auto', 'border':'1px solid #dee2e6'},
                                 style_cell={'textAlign': 'left', 'padding': '10px'},
                                 style_header={'backgroundColor': PRIMARY_COLOR, 'fontWeight': 'bold', 'color': 'white'},
                                 style_data_conditional=[
                                     {'if': {'column_id': 'salary_resid', 'filter_query': '{salary_resid} < 0'},
                                      'backgroundColor': '#ffe6e6', 'color': SEMANTIC_RED}
                                 ])
        ], className='col-md-6')
    ], className='row'),

    html.Div(style={'height':'30px'})
], className='container-fluid', style={'fontFamily':'Arial, sans-serif'})

# -----------------------------
# Callbacks
# -----------------------------
# Callback 1: Update role options 
@app.callback(
    Output('role-filter', 'options'),
    Output('role-filter', 'value'),
    Output('role-filter', 'disabled'),
    Input('dept-filter', 'value'),
    State('role-filter', 'value')
)
def update_role_options(dept_values, current_role_value):
    if not dept_values:
        return [], None, True 

    if not isinstance(dept_values, list):
        dept_values = [dept_values]

    roles_in_selected = sorted(df[df['Department'].isin(dept_values)]['Role'].dropna().unique().tolist())
    options = [{'label': r, 'value': r} for r in roles_in_selected]
    
    new_value = None
    if current_role_value:
        if not isinstance(current_role_value, list):
            current_role_value = [current_role_value]
        
        preserved = [r for r in current_role_value if r in roles_in_selected]
        
        new_value = preserved if preserved else None

    return options, new_value, False 

# Callback 2: Main update 
@app.callback(
    Output('scatter-salary-perf','figure'),
    Output('dept-summary','figure'),
    Output('violin-salary','figure'),
    Output('resid-violin','figure'),
    Output('resid-hist','figure'),
    Output('underpaid-table','data'),
    Output('kpis','children'),
    Input('dept-filter','value'),
    Input('role-filter','value'),
    Input('cutoff-quantile','value'),
    Input('perf-min','value')
)
def update_all(dept_values, role_values, cutoff_q, perf_min):
    dff = df.copy()

    # 1. Department filter
    if dept_values:
        if not isinstance(dept_values, list): dept_values = [dept_values]
        dff = dff[dff['Department'].isin(dept_values)]

    # 2. Role filter
    if role_values:
        if not isinstance(role_values, list): role_values = [role_values]
        valid_roles = dff['Role'].unique().tolist()
        sel_roles = [r for r in role_values if r in valid_roles]
        if sel_roles:
            dff = dff[dff['Role'].isin(sel_roles)]

    # 3. Performance minima filter
    perf_cut = perf_min if perf_min is not None else DEFAULT_PERF_CUT
    
    if not dff.empty and perf_cut is not None:
        dff_filtered_perf = dff[dff['PerformanceRating'] >= perf_cut].copy()
    else:
         dff_filtered_perf = dff.copy()

    # Recompute underpaid list 
    up, thr = compute_underpaid(dff_filtered_perf, quantile=cutoff_q, perf_cut=perf_cut)
    
    # Save the filtered underpaid list 
    cols_to_save = [c for c in ['EmployeeID','Name','Department','Role','ManagerID','PerformanceRating','AnnualSalaryINR','salary_resid'] if c in up.columns]
    if not up.empty:
        up.loc[:, cols_to_save].to_csv(UNDERPAID_CSV, index=False)
    else:
        pd.DataFrame(columns=cols_to_save).to_csv(UNDERPAID_CSV, index=False)

    # Figures
    fig1 = fig_salary_vs_perf(dff_filtered_perf)
    fig2 = fig_dept_summary(dff_filtered_perf)
    fig3 = fig_salary_violin(dff_filtered_perf)
    fig4 = fig_residuals_violin(dff_filtered_perf, thr)
    fig5 = fig_resid_hist(dff_filtered_perf, cutoff_q)

    # Table data and KPIs
    table_cols = ['EmployeeID','Name','Department','Role','ManagerID','PerformanceRating','AnnualSalaryINR','salary_resid']
    table_data = up[[c for c in table_cols if c in up.columns]].to_dict('records') 
    kpis_div = kpi_cards(dff_filtered_perf, up)
    
    return fig1, fig2, fig3, fig4, fig5, table_data, kpis_div

# -----------------------------
# Run app 
# -----------------------------
if __name__ == '__main__':
    print("Starting Dash app on http://127.0.0.1:8050")
    # Use app.run_server for robust deployment
    app.run(debug=False)