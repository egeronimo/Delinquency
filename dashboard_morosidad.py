# =============================================
# IMPORT LIBRARIES
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import requests
from io import BytesIO
import tempfile
import os
warnings.filterwarnings('ignore')
from datetime import timedelta

# =============================================
# PAGE CONFIGURATION
# =============================================
st.set_page_config(
    page_title="Delinquency Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================
# HELPER FUNCTIONS
# =============================================
def format_currency(value):
    """Format a numeric value as currency with separators"""
    try:
        num = float(value)
        if pd.isna(num):
            return "N/A"
        return "${:,.2f}".format(num) if num % 1 else "${:,.0f}".format(num)
    except (ValueError, TypeError):
        return str(value)

def format_percent(value):
    """Format a value as percentage"""
    try:
        num = float(value)
        if pd.isna(num):
            return "N/A"
        return "{:.1%}".format(num)
    except (ValueError, TypeError):
        return str(value)

# =============================================
# DATA LOADING AND PROCESSING
# =============================================
@st.cache_data
def load_data():
    try:
        # Google Sheets file (shared as "anyone with the link")
        file_id = "1G5vuN9pSVcsc1UcMKXXHM21Kwjr8SU__"
        drive_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"

        with st.spinner('Loading data...'):
            # Leer directamente el Excel desde Google Drive (sin autenticación)
            estado_cuenta = pd.read_excel(drive_url, sheet_name='estado_de_cuenta', dtype={'NCF': str, 'Documento': str})
            comportamiento_pago = pd.read_excel(drive_url, sheet_name='comportamiento_de_pago', dtype={'NCF': str, 'Documento': str})

        # Data cleaning and transformation
        estado_cuenta['Fecha_fatura'] = pd.to_datetime(estado_cuenta['Fecha_fatura'], errors='coerce')
        estado_cuenta['Fecha_vencimiento'] = pd.to_datetime(estado_cuenta['Fecha_vencimiento'], errors='coerce')

        if 'Dias' not in estado_cuenta.columns:
            estado_cuenta['Dias'] = (pd.to_datetime('today') - estado_cuenta['Fecha_vencimiento']).dt.days

        amount_cols = ['Inicial', 'Balance', '0-30 Dias', '31-60 Dias', '61-90 Dias', '91-120 Dias', 'Mas 120 Dias']
        for col in amount_cols:
            if col in estado_cuenta.columns:
                estado_cuenta[col] = pd.to_numeric(estado_cuenta[col].astype(str).str.replace(',', ''), errors='coerce')

        # Clasificación morosidad
        estado_cuenta['Estado_Morosidad'] = np.where(
            estado_cuenta['Dias'] > 120, 'Severe Delinquency (+120 days)',
            np.where(
                estado_cuenta['Dias'] > 90, 'High Delinquency (91-120 days)',
                np.where(
                    estado_cuenta['Dias'] > 60, 'Moderate Delinquency (61-90 days)',
                    np.where(
                        estado_cuenta['Dias'] > 30, 'Early Warning (31-60 days)',
                        'Current (0-30 days)'
                    )
                )
            )
        )

        # Procesar comportamiento de pago
        comportamiento_pago['Fecha_fatura'] = pd.to_datetime(comportamiento_pago['Fecha_fatura'], errors='coerce')
        amount_cols_pago = ['Pagado', 'Descuento', 'Total', 'Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']
        for col in amount_cols_pago:
            if col in comportamiento_pago.columns:
                comportamiento_pago[col] = pd.to_numeric(comportamiento_pago[col].astype(str).str.replace(',', ''), errors='coerce')

        return estado_cuenta, comportamiento_pago

    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# =============================================
# LOAD DATA
# =============================================
estado_cuenta, comportamiento_pago = load_data()

# Verify if data loaded correctly
if estado_cuenta.empty or comportamiento_pago.empty:
    st.error("Could not load data. Please check the file path and structure.")
    st.stop()

# =============================================
# PREDICTIVE MODEL
# =============================================
try:
    # Prepare data for model
    X = estado_cuenta[['Dias', 'Inicial', 'Balance']].fillna(0)
    y = (estado_cuenta['Dias'] > 60).astype(int)  # Delinquency based on definition
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    # Add predictions
    estado_cuenta['Probabilidad_Morosidad'] = model.predict_proba(X)[:, 1]
    
    # Corrected segmentation with 4 categories
    estado_cuenta['Segmento_Riesgo'] = pd.cut(
        estado_cuenta['Probabilidad_Morosidad'],
        bins=[0, 0.3, 0.6, 0.8, 1],
        labels=['Low (0-30%)', 'Moderate (30-60%)', 'High (60-80%)', 'Extreme (80-100%)'],
        include_lowest=True
    )
    
except Exception as e:
    st.warning(f"Predictive model could not be trained: {str(e)}")
    estado_cuenta['Probabilidad_Morosidad'] = np.where(
        estado_cuenta['Dias'] > 60, 0.85,
        np.where(
            estado_cuenta['Dias'] > 30, 0.5,
            np.where(
                estado_cuenta['Dias'] > 15, 0.3,
                0.1
            )
        )
    )
    estado_cuenta['Segmento_Riesgo'] = pd.cut(
        estado_cuenta['Probabilidad_Morosidad'],
        bins=[0, 0.3, 0.6, 0.8, 1],
        labels=['Low (0-30%)', 'Moderate (30-60%)', 'High (60-80%)', 'Extreme (80-100%)']
    )

# =============================================
# DASHBOARD INTERFACE
# =============================================
st.title("📊 Delinquency Analysis Dashboard")
st.markdown("""
    **Analysis** of payment behavior, delinquency, and customer credit risk.
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📌 Executive Summary", 
    "🔍 Delinquency Analysis",
    "🔮 Risk Prediction",
    "👤 Customer Profile",
    "🧩 Segmentation",
    "🧮 Simulator",
    "💵 Collection Goals & Compliance",
    "📊 Credit Limit by Customer"
])

# =============================================
# TAB 1: EXECUTIVE SUMMARY
# =============================================
with tab1:
    st.header("📌 Executive Summary", divider="blue")
    
    # MAIN KPIs
    total_cartera = estado_cuenta['Balance'].sum()
    total_morosidad = estado_cuenta[estado_cuenta['Dias'] > 30]['Balance'].sum()
    clientes_morosos = estado_cuenta[estado_cuenta['Dias'] > 30]['Codigo'].nunique()
    porcentaje_morosidad = (total_morosidad / total_cartera) if total_cartera > 0 else 0
    dso = (estado_cuenta['Balance'] * estado_cuenta['Dias']).sum() / total_cartera if total_cartera > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Portfolio", f"${total_cartera:,.2f}")
    with col2:
        st.metric("⚠️ Total Delinquency", 
                f"${total_morosidad:,.2f}", 
                f"{porcentaje_morosidad:.1%} of portfolio")
    with col3:
        st.metric("👥 Delinquent Customers", clientes_morosos)
    with col4:
        st.metric("⏳ Average DSO", f"{dso:.0f} days")
    
    # TREND CHARTS
    st.subheader("📈 Time Evolution", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        # Monthly evolution chart
        estado_cuenta['Mes'] = estado_cuenta['Fecha_fatura'].dt.to_period('M').astype(str)
        evolucion_mensual = estado_cuenta.groupby('Mes')['Balance'].sum().reset_index()
        
        fig = px.line(
            evolucion_mensual,
            x='Mes',
            y='Balance',
            title='Monthly Portfolio Evolution',
            labels={'Balance': 'Balance ($)', 'Mes': 'Period'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution by delinquency status
        distrib_morosidad = estado_cuenta.groupby('Estado_Morosidad')['Balance'].sum().reset_index()
        
        fig = px.pie(
            distrib_morosidad,
            names='Estado_Morosidad',
            values='Balance',
            title='Distribution by Delinquency Status',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # PAYMENT METHOD ANALYSIS (CORRECTED VERSION)
    st.subheader("💳 Payment Methods Analysis", divider="gray")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not comportamiento_pago.empty:
            payment_methods = comportamiento_pago[['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']].sum().reset_index()
            payment_methods.columns = ['Method', 'Amount']
            fig = px.pie(
                payment_methods,
                names='Method',
                values='Amount',
                title='Payment Methods Distribution',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not comportamiento_pago.empty and 'Pagado' in comportamiento_pago.columns:
            # Payment methods summary
            payment_methods = ['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']
            summary = []
            
            for method in payment_methods:
                if method in comportamiento_pago.columns:
                    # Clean specific method
                    comportamiento_pago[method] = (
                        comportamiento_pago[method]
                        .astype(str)
                        .str.replace(',', '')
                        .apply(pd.to_numeric, errors='coerce')
                        .fillna(0)
                    )
                    
                    # Filter only transactions with this method
                    mask = (comportamiento_pago[method] > 0)
                    total_paid = comportamiento_pago.loc[mask, 'Pagado'].sum()
                    num_transactions = mask.sum()
                    
                    summary.append({
                        'Method': method,
                        'Total Paid': total_paid,
                        'Transactions': num_transactions
                    })
            
            # Create and display table
            df_summary = pd.DataFrame(summary)
            
            st.dataframe(
                df_summary.assign(
                    **{
                        'Total Paid': df_summary['Total Paid'].apply(lambda x: f"${x:,.2f}"),
                        '% of Total': df_summary['Total Paid'] / df_summary['Total Paid'].sum()
                    }
                ).sort_values('Total Paid', ascending=False),
                column_config={
                    '% of Total': st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=1
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Additional statistics
            with st.expander("📌 Detailed Statistics"):
                st.write(f"**Total paid:** ${comportamiento_pago['Pagado'].sum():,.2f}")
                st.write(f"**Average payment:** ${comportamiento_pago['Pagado'].mean():,.2f}")
                st.write(f"**Recorded transactions:** {len(comportamiento_pago)}")
                st.write(f"**Unique customers:** {comportamiento_pago['Codigo'].nunique()}")
    
    # ADDITIONAL ANALYSIS
    st.subheader("🔍 Complementary Analysis", divider="gray")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 customers by balance
        top_clientes = (
            estado_cuenta.groupby('Nombre Cliente')
            .agg({'Balance': 'sum', 'Dias': 'mean'})
            .nlargest(10, 'Balance')
            .reset_index()
        )
        
        fig = px.bar(
            top_clientes,
            x='Nombre Cliente',
            y='Balance',
            title='Top 10 Customers by Balance',
            labels={'Balance': 'Balance ($)', 'Nombre Cliente': 'Customer'},
            hover_data=['Dias']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Days overdue distribution
        fig = px.box(
            estado_cuenta,
            x='Dias',
            title='Days Overdue Distribution',
            labels={'Dias': 'Days overdue'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💳 Detailed Payment Methods Analysis", divider="gray")
    
    if not comportamiento_pago.empty and 'Pagado' in comportamiento_pago.columns:
        # Data cleaning
        comportamiento_pago['Pagado'] = (
            comportamiento_pago['Pagado']
            .astype(str)
            .str.replace(',', '')
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0)
        )
        # Prepare data for table
        payment_methods = ['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']
        results = []
        
        for method in payment_methods:
            if method in comportamiento_pago.columns:
                # Clean specific method
                comportamiento_pago[method] = (
                    comportamiento_pago[method]
                    .astype(str)
                    .str.replace(',', '')
                    .apply(pd.to_numeric, errors='coerce')
                    .fillna(0)
                )
                # Filter only transactions with this method
                mask_method = (comportamiento_pago[method] > 0)
                df_method = comportamiento_pago[mask_method].copy()
                
                # Get delinquency status if exists
                if not estado_cuenta.empty:
                    df_method = df_method.merge(
                        estado_cuenta[['Codigo', 'Estado_Morosidad']].drop_duplicates(),
                        on='Codigo',
                        how='left'
                    )
                    df_method['Estado_Morosidad'] = df_method['Estado_Morosidad'].fillna('No data')
                else:
                    df_method['Estado_Morosidad'] = 'No data'
                
                # Group by delinquency status
                group = df_method.groupby('Estado_Morosidad').agg({
                    'Pagado': 'sum',
                    'Codigo': 'nunique',
                    method: 'count'
                }).reset_index()
                
                for _, row in group.iterrows():
                    results.append({
                        'Method': method,
                        'Delinquency Status': row['Estado_Morosidad'],
                        'Total Paid': row['Pagado'],
                        'Transactions': row[method],
                        'Customers': row['Codigo']
                    })
        
        # Create final DataFrame
        if results:
            df_results = pd.DataFrame(results)
            total_general = df_results['Total Paid'].sum()
            
            # Format table
            st.dataframe(
                df_results.assign(
                    **{
                        'Total Paid': df_results['Total Paid'].apply(lambda x: f"${x:,.2f}"),
                        '% of Total': df_results['Total Paid'] / total_general
                    }
                ).sort_values(['Method', 'Total Paid'], ascending=[True, False]),
                column_config={
                    '% of Total': st.column_config.ProgressColumn(
                        format="%.1f%%",
                        min_value=0,
                        max_value=1,
                        width="medium"
                    )
                },
                hide_index=True,
                use_container_width=True,
                height=500
            )
            
# =============================================
# TAB 2: DELINQUENCY ANALYSIS
# =============================================
with tab2:
    st.header("🔍 Detailed Delinquency Analysis", divider="blue")
    
    st.subheader("🔎 Delinquency Distribution", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        # Delinquency heatmap by days range
        morosidad_rangos = estado_cuenta[['0-30 Dias', '31-60 Dias', '61-90 Dias', '91-120 Dias', 'Mas 120 Dias']].sum().reset_index()
        morosidad_rangos.columns = ['Range', 'Amount']
        fig = px.bar(
            morosidad_rangos,
            x='Range',
            y='Amount',
            title='Total by Days Range of Delinquency',
            color='Range'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top 10 delinquent customers
        top_morosos = estado_cuenta[estado_cuenta['Dias'] > 60].groupby('Nombre Cliente')['Balance'].sum().nlargest(10).reset_index()
        fig = px.bar(
            top_morosos,
            x='Nombre Cliente',
            y='Balance',
            title='Top 10 Customers with Highest Delinquency',
            labels={'Balance': 'Delinquency Amount', 'Nombre Cliente': 'Customer'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📅 Due Date Behavior", divider="gray")
    
    # Analysis by day of week
    estado_cuenta['Dia_Semana'] = estado_cuenta['Fecha_vencimiento'].dt.day_name()
    estado_cuenta['Mes'] = estado_cuenta['Fecha_vencimiento'].dt.month_name()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            estado_cuenta.groupby('Dia_Semana')['Balance'].sum().reset_index(),
            x='Dia_Semana',
            y='Balance',
            title='Delinquency by Due Date Day of Week',
            category_orders={"Dia_Semana": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            estado_cuenta.groupby('Mes')['Balance'].sum().reset_index(),
            x='Mes',
            y='Balance',
            title='Delinquency by Due Date Month',
            category_orders={"Mes": ["January", "February", "March", "April", "May", "June", "July",
                                   "August", "September", "October", "November", "December"]}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # New section: Delinquency Concentration
    st.subheader("🎯 Delinquency Concentration", divider="gray")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pareto analysis (80/20)
        clientes_ordenados = estado_cuenta.groupby('Nombre Cliente')['Balance'].sum().sort_values(ascending=False)
        clientes_ordenados = (clientes_ordenados.cumsum() / clientes_ordenados.sum() * 100).reset_index()
        clientes_ordenados['Es80'] = clientes_ordenados['Balance'] <= 80
        
        fig = px.bar(
            clientes_ordenados,
            x='Nombre Cliente',
            y='Balance',
            color='Es80',
            title='80/20 Rule - Cumulative % of Delinquency',
            labels={'Balance': 'Cumulative %', 'Nombre Cliente': 'Customers ordered by delinquency'}
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Days overdue analysis
        fig = px.box(
            estado_cuenta,
            x='Estado_Morosidad',
            y='Dias',
            title='Days Overdue Distribution by Category',
            points="all"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # New section: Time Trends
    st.subheader("📅 Delinquency Time Evolution", divider="gray")
    
    # Group by month and delinquency category
    estado_cuenta['Mes'] = estado_cuenta['Fecha_fatura'].dt.to_period('M').astype(str)
    evolucion_morosidad = estado_cuenta.groupby(['Mes', 'Estado_Morosidad'])['Balance'].sum().unstack().fillna(0)
    
    fig = px.line(
        evolucion_morosidad,
        title='Monthly Delinquency Evolution',
        labels={'value': 'Amount', 'variable': 'Delinquency Status'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Explanation of new analyses
    st.markdown("""
    **🔍 What do these new analyses provide?**
    - **80/20 Rule**: Identifies if 20% of customers concentrate 80% of delinquency (Pareto principle).
    - **Box plot**: Shows days overdue dispersion in each category, revealing hidden patterns.
    - **Time evolution**: Allows detecting if delinquency is improving or worsening over time.
    """)

# =============================================
# TAB 3: RISK PREDICTION
# =============================================
with tab3:
    st.header("🔮 Delinquency Risk Prediction", divider="blue")
    
    st.subheader("📊 Factors Influencing Delinquency", divider="gray")
    
    if 'model' in locals():
        feature_importance = pd.DataFrame({
            'Variable': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Variable',
            y='Importance',
            title='Variable Importance in Model',
            labels={'Variable': 'Factor', 'Importance': 'Relative Importance'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - **Days**: Main risk indicator (more days = higher delinquency probability)
        - **Balance**: Higher amounts usually associated with higher risk
        - **Initial**: Relationship with customer's initial payment capacity
        """)
    else:
        st.warning("Predictive model not available. Showing simulated data.")
    
    st.subheader("🧑‍💼 Customer Profiles by Risk Level", divider="gray")
    
    col1, col2 = st.columns(2)
    with col1:
        risk_profiles = estado_cuenta.groupby('Segmento_Riesgo').agg({
            'Dias': 'mean',
            'Balance': 'sum',
            'Codigo': 'nunique',
            'Probabilidad_Morosidad': 'mean'
        }).reset_index()
        
        st.dataframe(
            risk_profiles.assign(
                Balance=risk_profiles['Balance'].apply(format_currency),
                Dias=risk_profiles['Dias'].apply(lambda x: f"{x:.0f}"),
                Probabilidad_Morosidad=risk_profiles['Probabilidad_Morosidad'].apply(format_percent)
            ).rename(columns={
                'Codigo': 'Qty. Customers',
                'Probabilidad_Morosidad': 'Avg. Risk'
            }),
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        fig = px.box(
            estado_cuenta,
            x='Segmento_Riesgo',
            y='Dias',
            color='Segmento_Riesgo',
            title='Days Overdue Distribution by Segment',
            labels={'Dias': 'Days overdue'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🛡️ Action Plan by Risk Level", divider="gray")
    
    st.markdown("""
    | Risk Level | Recommended Actions | Follow-up Frequency |
    |------------|---------------------|---------------------|
    | **Low (0-30%)** | Automatic credit renewal | Quarterly |
    | **Moderate (30-60%)** | Additional verification, reduced limit | Monthly |
    | **High (60-80%)** | Guarantees required, advance payments | Weekly |
    | **Extreme (80-100%)** | Preventive collection, cash payment | Daily |
    """)

# =============================================
# TAB 4: CUSTOMER PROFILE
# =============================================
with tab4:
    st.header("👤 Customer Profile", divider="blue")
    
    # Get all unique customers from BOTH data sources
    clientes_estado = estado_cuenta[['Codigo', 'Nombre Cliente']].drop_duplicates()
    clientes_pago = comportamiento_pago[['Codigo', 'Nombre Cliente']].drop_duplicates()
    
    # Combine both DataFrames and remove duplicates
    todos_clientes = pd.concat([clientes_estado, clientes_pago]).drop_duplicates('Codigo')
    
    # Create options for selectbox
    opciones_clientes = {
        row['Codigo']: f"{row['Codigo']} - {row['Nombre Cliente']}" 
        for _, row in todos_clientes.iterrows()
    }
    
    cliente_seleccionado = st.selectbox(
        "🔍 Search Customer by Code or Name",
        options=list(opciones_clientes.keys()),
        format_func=lambda x: opciones_clientes[x]
    )
    
    # Get selected customer data from both sources
    cliente_data_estado = estado_cuenta[estado_cuenta['Codigo'] == cliente_seleccionado]
    cliente_data_pago = comportamiento_pago[comportamiento_pago['Codigo'] == cliente_seleccionado]
    
    # Determine customer name (can come from either source)
    cliente_nombre = ""
    if not cliente_data_estado.empty:
        cliente_nombre = cliente_data_estado['Nombre Cliente'].iloc[0]
    elif not cliente_data_pago.empty:
        cliente_nombre = cliente_data_pago['Nombre Cliente'].iloc[0]
    else:
        st.error("No information found for this customer")
        st.stop()
    
    st.subheader(f"📋 General Information for {cliente_nombre}", divider="gray")
    
    # Show different metrics depending on available data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not cliente_data_estado.empty:
            st.metric("📅 Pending Invoices", len(cliente_data_estado))
            st.metric("💰 Total Balance", format_currency(cliente_data_estado['Balance'].sum()))
        else:
            st.metric("📅 Pending Invoices", "N/A")
            st.metric("💰 Total Balance", "N/A")
    
    with col2:
        if not cliente_data_estado.empty:
            st.metric("⏱️ Average Days Overdue", f"{cliente_data_estado['Dias'].mean():.0f}")
            morosidad = (cliente_data_estado[cliente_data_estado['Dias'] > 60]['Balance'].sum() / 
                        cliente_data_estado['Balance'].sum() * 100) if cliente_data_estado['Balance'].sum() > 0 else 0
            st.metric("📉 Delinquency Percentage", f"{morosidad:.1f}%")
        else:
            st.metric("⏱️ Average Days Overdue", "N/A")
            st.metric("📉 Delinquency Percentage", "N/A")
    
    with col3:
        if not cliente_data_estado.empty:
            st.metric("⚠️ Delinquency Probability", 
                     format_percent(cliente_data_estado['Probabilidad_Morosidad'].mean()))
            st.metric("💳 Recommended Limit", 
                     format_currency(cliente_data_estado['Inicial'].mean() * 0.8))
        else:
            st.metric("⚠️ Delinquency Probability", "N/A")
            st.metric("💳 Recommended Limit", "N/A")
    
    st.subheader("📅 Payment History", divider="gray")
    
    if not cliente_data_pago.empty:
        fig = px.line(
            cliente_data_pago.sort_values('Fecha_fatura'), 
            x='Fecha_fatura', 
            y='Pagado',
            title='Payment History',
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        payment_methods = cliente_data_pago[['Efectivo', 'Cheque', 'Tarjeta', 'Transferencia']].sum().reset_index()
        payment_methods.columns = ['Method', 'Amount']
        fig = px.pie(
            payment_methods, 
            names='Method', 
            values='Amount',
            title='Payment Methods Used',
            hole=0.3
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show table with last 5 payments
        st.write("**Last Payments Made:**")
        st.dataframe(
            cliente_data_pago.sort_values('Fecha_fatura', ascending=False).head(5)[[
                'Fecha_fatura', 'Pagado', 'Efectivo', 'Cheque', 'Tarjeta', 'Transferencia'
            ]].rename(columns={
                'Fecha_fatura': 'Date',
                'Pagado': 'Total Paid'
            }),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No payment history found for this customer")
    
    # Show message if no account status data
    if cliente_data_estado.empty:
        st.info("ℹ️ This customer has no pending invoices in the account status")

# =============================================
# TAB 5: SEGMENTATION
# =============================================
with tab5:
    st.header("🧩 Customer Segmentation", divider="blue")
    
    st.subheader("🔢 Clustering by Behavior", divider="gray")
    
    # Prepare data for clustering
    cluster_data = estado_cuenta.groupby('Codigo').agg({
        'Nombre Cliente': 'first',
        'Dias': 'mean',
        'Balance': 'sum',
        'Probabilidad_Morosidad': 'mean',
        'Estado_Morosidad': lambda x: x.value_counts().index[0] if not x.empty else 'No data'
    }).reset_index()
    
    # Select features for clustering
    features = ['Dias', 'Balance', 'Probabilidad_Morosidad']
    
    # Step 1: Handle missing values
    imputer = SimpleImputer(strategy='median')
    cluster_data[features] = imputer.fit_transform(cluster_data[features])
    
    # Step 2: Normalize data (except Probabilidad_Morosidad which is already 0-1)
    scaler = StandardScaler()
    cluster_data[['Dias', 'Balance']] = scaler.fit_transform(cluster_data[['Dias', 'Balance']])
    
    # Verify no NaN after processing
    if cluster_data[features].isna().any().any():
        st.warning("Warning: There are still missing values in the data. Those rows will be removed.")
        cluster_data = cluster_data.dropna(subset=features)
    
    # Apply K-Means only if enough data
    if len(cluster_data) >= 4:
        try:
            kmeans = KMeans(n_clusters=4, random_state=42)
            cluster_data['Cluster'] = kmeans.fit_predict(cluster_data[features])
            
            # Prepare size for chart (ensure positive values)
            cluster_data['Size'] = cluster_data['Probabilidad_Morosidad'] * 20 + 5  # Scale to range 5-25
            
            # Cluster visualization
            fig = px.scatter(
                cluster_data,
                x='Dias',
                y='Balance',
                color='Cluster',
                size='Size',  # Use column with positive values
                hover_data=['Nombre Cliente', 'Probabilidad_Morosidad'],
                title='Customer Segmentation by Behavior',
                labels={
                    'Dias': 'Days Overdue (normalized)',
                    'Balance': 'Total Balance (normalized)',
                    'Cluster': 'Group',
                    'Probabilidad_Morosidad': 'Delinquency Risk'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster description
            st.subheader("📝 Group Descriptions", divider="gray")
            
            # Calculate statistics by cluster
            cluster_stats = cluster_data.groupby('Cluster').agg({
                'Dias': ['mean', 'std'],
                'Balance': ['sum', 'mean'],
                'Probabilidad_Morosidad': 'mean',
                'Codigo': 'nunique'
            }).reset_index()
            
            # Rename columns
            cluster_stats.columns = [
                'Cluster',
                'Avg_Days', 'Std_Days',
                'Total_Balance', 'Avg_Balance',
                'Avg_Risk',
                'Customer_Count'
            ]
            
            # Convert normalized values to approximate original scale
            cluster_stats[['Avg_Days', 'Avg_Balance']] = scaler.inverse_transform(
                cluster_stats[['Avg_Days', 'Avg_Balance']])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(
                    cluster_stats.assign(
                        Total_Balance=cluster_stats['Total_Balance'].apply(format_currency),
                        Avg_Balance=cluster_stats['Avg_Balance'].apply(format_currency),
                        Avg_Risk=cluster_stats['Avg_Risk'].apply(format_percent),
                        Avg_Days=cluster_stats['Avg_Days'].apply(lambda x: f"{x:.0f}")
                    ),
                    hide_index=True,
                    use_container_width=True
                )
            
            with col2:
                st.markdown("""
                **Group Characteristics:**
                
                1. **Group 0 (Punctual Customers)**:
                   - Days overdue: 0-15
                   - Average balance: $1K-5K
                   - Delinquency risk: 0-20%
                
                2. **Group 1 (Moderate Customers)**:
                   - Days overdue: 16-45
                   - Average balance: $5K-15K
                   - Delinquency risk: 20-40%
                
                3. **Group 2 (High Risk Customers)**:
                   - Days overdue: 46-90
                   - Average balance: $15K-50K
                   - Delinquency risk: 40-60%
                
                4. **Group 3 (Critical Customers)**:
                   - Days overdue: 90+
                   - Average balance: $50K+
                   - Delinquency risk: 60-100%
                """)
            
        except Exception as e:
            st.error(f"Error in clustering: {str(e)}")
    else:
        st.warning("Not enough data for clustering (at least 4 customers needed).")

# =============================================
# TAB 6: SIMULATOR
# =============================================
with tab6:
    st.header("🧮 Credit Risk Simulator", divider="blue")
    
    # Get all unique customers from BOTH data sources
    clientes_estado = estado_cuenta[['Codigo', 'Nombre Cliente']].drop_duplicates()
    clientes_pago = comportamiento_pago[['Codigo', 'Nombre Cliente']].drop_duplicates()
    todos_clientes = pd.concat([clientes_estado, clientes_pago]).drop_duplicates('Codigo')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create options for selectbox
        opciones_clientes = {
            row['Codigo']: f"{row['Codigo']} - {row['Nombre Cliente']}" 
            for _, row in todos_clientes.iterrows()
        }
        
        cliente_sim = st.selectbox(
            "👤 Select Customer for Simulation",
            options=list(opciones_clientes.keys()),
            format_func=lambda x: opciones_clientes[x]
        )
        
        # Get customer data from estado_cuenta (if exists)
        cliente_info_estado = estado_cuenta[estado_cuenta['Codigo'] == cliente_sim]
        
        # Get customer data from comportamiento_pago (if exists)
        cliente_info_pago = comportamiento_pago[comportamiento_pago['Codigo'] == cliente_sim]
        
        # Determine customer name
        cliente_nombre = ""
        if not cliente_info_estado.empty:
            cliente_nombre = cliente_info_estado['Nombre Cliente'].iloc[0]
        elif not cliente_info_pago.empty:
            cliente_nombre = cliente_info_pago['Nombre Cliente'].iloc[0]
        
        # Show metrics with default values if no estado_cuenta data
        balance_actual = cliente_info_estado['Balance'].sum() if not cliente_info_estado.empty else 0
        dias_atraso = cliente_info_estado['Dias'].mean() if not cliente_info_estado.empty else 0
        riesgo_actual = cliente_info_estado['Probabilidad_Morosidad'].mean() if not cliente_info_estado.empty else 0.3  # Default value
        
        st.metric("📊 Current Balance", format_currency(balance_actual))
        st.metric("⏳ Days Overdue", f"{dias_atraso:.0f}")
        st.metric("⚠️ Current Risk", format_percent(riesgo_actual))
        
        # Get initial amount for simulation (use payment average if no estado_cuenta)
        if not cliente_info_estado.empty:
            monto_inicial = cliente_info_estado['Inicial'].mean()
        elif not cliente_info_pago.empty:
            monto_inicial = cliente_info_pago['Pagado'].mean()
        else:
            monto_inicial = 10000  # Default value
    
    with col2:
        monto_simular = st.number_input(
            "💰 Amount to Evaluate",
            min_value=0.0,
            value=float(monto_inicial),
            step=1000.0
        )
        
        plazo_simular = st.number_input(
            "📅 Term to Evaluate (days)",
            min_value=1,
            value=30,
            step=15
        )
        
        # Additional factors for simulation
        st.markdown("**🔍 Additional Factors**")
        
        # Calculate payment history based on available data
        if not cliente_info_pago.empty:
            # If we have payment data, calculate score based on:
            # 1. Total number of payments
            # 2. Amount consistency
            # 3. Payment frequency
            
            num_pagos = cliente_info_pago.shape[0]
            monto_std = cliente_info_pago['Pagado'].std()
            freq_pagos = (cliente_info_pago['Fecha_fatura'].max() - cliente_info_pago['Fecha_fatura'].min()).days / num_pagos if num_pagos > 1 else 30
            
            # Calculate score (1-5)
            historial_valor = min(5, max(1, 
                int((num_pagos/10) +          # More payments = better
                (1 - monto_std/monto_inicial if monto_inicial > 0 else 0) +  # Less variation = better
                (30/freq_pagos if freq_pagos > 0 else 1)    # More frequent payments = better
            )))
        else:
            historial_valor = 3  # Default value if no data
            
        historial_pagos = st.slider(
            "Payment History (1 = Poor, 5 = Excellent)",
            min_value=1,
            max_value=5,
            value=historial_valor
        )
        
        # Determine customer type based on available data
        if not cliente_info_pago.empty:
            num_pagos = cliente_info_pago.shape[0]
            tipo_index = min(3, int(num_pagos / 5))  # More payments = better classification
        else:
            tipo_index = 0  # "New" by default
            
        tipo_cliente = st.selectbox(
            "Customer Type",
            options=["New", "Occasional", "Recurrent", "Preferred"],
            index=tipo_index
        )
        
        # Improved simulated risk calculation
        riesgo_base = riesgo_actual
        
        # Adjust base risk based on payment history (if exists)
        if not cliente_info_pago.empty:
            # Calculate risk based on payment variability
            pagos_std = cliente_info_pago['Pagado'].std()
            riesgo_base = max(riesgo_base, min(0.7, pagos_std/(monto_inicial + 1e-6)))  # More variation = higher risk
            
        # Adjustment factors
        factor_monto = min(monto_simular / (monto_inicial + 1e-6), 3)  # Avoid division by zero
        factor_plazo = plazo_simular / 30
        factor_historial = 1.5 - (historial_pagos * 0.1)  # Better history reduces risk
        factor_tipo = {
            "New": 1.2,
            "Occasional": 1.1,
            "Recurrent": 0.9,
            "Preferred": 0.8
        }[tipo_cliente]
        
        # Final risk calculation
        riesgo_simulado = min(riesgo_base * factor_monto * factor_plazo * factor_historial * factor_tipo, 0.99)
        
        st.subheader("📈 Simulation Result", divider="gray")
        
        # Show result with style based on risk level
        if riesgo_simulado > 0.8:
            st.error(f"🚨 **High Risk** ({format_percent(riesgo_simulado)})")
        elif riesgo_simulado > 0.6:
            st.warning(f"⚠️ **High Risk** ({format_percent(riesgo_simulado)})")
        elif riesgo_simulado > 0.4:
            st.info(f"🔍 **Moderate Risk** ({format_percent(riesgo_simulado)})")
        else:
            st.success(f"✅ **Low Risk** ({format_percent(riesgo_simulado)})")
    
    st.subheader("📋 Detailed Recommendations", divider="gray")
    
    # Dynamic recommendations section
    if riesgo_simulado > 0.8:
        st.error("""
        **🚨 Recommended Actions:**
        1. Do not approve additional credit without solid guarantees
        2. Require advance payments (minimum 50%)
        3. Maximum term: 7 days
        4. Review complete payment history
        5. Consider preventive legal actions
        6. Reduce credit limit by 75%
        7. Daily case supervision
        """)
        
        # Show additional warning for customers without estado_cuenta history
        if cliente_info_estado.empty:
            st.warning("""
            **⚠️ Attention:** This customer has no pending invoices registered, 
            but risk analysis indicates high delinquency probability. 
            Carefully verify their external credit history.
            """)
            
    elif riesgo_simulado > 0.6:
        st.warning("""
        **⚠️ Recommended Actions:**
        1. Reduce credit limit by 50%
        2. Require guarantor or guarantee
        3. Maximum term: 15 days
        4. Weekly follow-up
        5. Early payment discounts (max 5%)
        6. Require recent banking history
        7. Automated early alerts
        """)
        
    elif riesgo_simulado > 0.4:
        st.info("""
        **🔍 Recommended Actions:**
        1. Maintain current limit or moderate increase (10-20%)
        2. Maximum term: 30 days
        3. Monthly follow-up
        4. Structured payment plans
        5. Automatic reminders at 5 and 2 days before
        6. Quarterly behavior review
        7. Consider guarantees for high amounts
        """)
        
    else:
        st.success("""
        **✅ Recommended Actions:**
        1. Consider line increase (20-30%)
        2. Flexible terms (up to 60 days)
        3. Semi-annual review
        4. Loyalty benefits (2% discount)
        5. Simplified approval process
        6. Include in preferred programs
        7. Evaluate special credits with preferential rates
        """)
    
    # Show additional information for customers without estado_cuenta data
    if cliente_info_estado.empty and not cliente_info_pago.empty:
        st.markdown("---")
        st.subheader("ℹ️ Additional Payment History Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Registered Payments", cliente_info_pago.shape[0])
            st.metric("Average Payment", format_currency(cliente_info_pago['Pagado'].mean()))
            
        with col2:
            ultimo_pago = cliente_info_pago['Fecha_fatura'].max()
            st.metric("Last Payment", ultimo_pago.strftime("%d/%m/%Y") if not pd.isnull(ultimo_pago) else "N/A")
            
            monto_total = cliente_info_pago['Pagado'].sum()
            st.metric("Total Amount Paid", format_currency(monto_total))

# =============================================
# TAB 7: COLLECTION GOALS & COMPLIANCE
# =============================================

with tab7:
    st.header("💵 Collection Goals & Compliance", divider="blue")

    # =======================
    # WEEK LIST
    # =======================
    hoy = pd.Timestamp.today().normalize()
    fechas_semana = pd.date_range(
        start=hoy - timedelta(weeks=3),
        end=hoy + timedelta(weeks=6),
        freq='W-MON'
    )

    semanas_unicas = pd.DataFrame({'fecha_min': fechas_semana})
    semanas_unicas['fecha_max'] = semanas_unicas['fecha_min'] + timedelta(days=6)
    semanas_unicas['Año'] = semanas_unicas['fecha_min'].dt.year
    semanas_unicas['Semana'] = semanas_unicas['fecha_min'].dt.isocalendar().week
    semanas_unicas['Etiqueta'] = semanas_unicas.apply(
        lambda row: f"Week {row['Semana']} ({row['fecha_min'].strftime('%d/%m/%Y')} - {row['fecha_max'].strftime('%d/%m/%Y')})",
        axis=1
    )

    semana_sel = st.selectbox("📅 Select target week for collection", options=semanas_unicas['Etiqueta'])
    semana_data = semanas_unicas[semanas_unicas['Etiqueta'] == semana_sel].iloc[0]
    fecha_ini_semana = semana_data['fecha_min']
    fecha_fin_semana = semana_data['fecha_max']
    es_futura = fecha_ini_semana > hoy
    st.info(f"📌 Selected week: {semana_sel}")

    # =======================
    # WEEKLY GOAL
    # =======================
    meta_semanal = st.number_input(
        "🎯 Weekly collection goal",
        min_value=1_200_000.0,
        max_value=1_600_000.0,
        value=1_400_000.0,
        step=50_000.0
    )

    # =======================
    # OPTIONAL INCLUDE OVERDUE
    # =======================
    incluir_atrasados = st.checkbox("📌 Include customers with overdue invoices from previous weeks", value=True)

    if incluir_atrasados:
        estado_cuenta_plan = estado_cuenta[estado_cuenta['Fecha_fatura'] <= fecha_fin_semana].copy()
    else:
        estado_cuenta_plan = estado_cuenta[
            (estado_cuenta['Fecha_fatura'] >= fecha_ini_semana) &
            (estado_cuenta['Fecha_fatura'] <= fecha_fin_semana)
        ].copy()

    # =======================
    # NORMALIZED PAYMENT HISTORY
    # =======================
    historial_pago = (
        comportamiento_pago.groupby("Codigo")["Pagado"]
        .mean()
        .reset_index()
        .rename(columns={"Pagado": "HistorialPago"})
    )
    max_hist = historial_pago["HistorialPago"].max() or 1
    historial_pago["HistorialPago_Normalizado"] = historial_pago["HistorialPago"] / max_hist

    # =======================
    # CALCULATE COLLECTION SCORE
    # =======================
    df_plan = (
        estado_cuenta_plan.groupby(["Codigo", "Nombre Cliente"])
        .agg({
            "Balance": "sum",
            "Probabilidad_Morosidad": "mean"
        })
        .reset_index()
    )

    max_balance = df_plan["Balance"].max() or 1
    df_plan["Balance_Normalizado"] = df_plan["Balance"] / max_balance
    df_plan = df_plan.merge(historial_pago[["Codigo", "HistorialPago_Normalizado"]], on="Codigo", how="left")
    df_plan["HistorialPago_Normalizado"] = df_plan["HistorialPago_Normalizado"].fillna(0.5)

    df_plan["Score_Cobro"] = (
        0.5 * (1 - df_plan["Probabilidad_Morosidad"]) +
        0.3 * df_plan["HistorialPago_Normalizado"] +
        0.2 * df_plan["Balance_Normalizado"]
    )

    # =======================
    # SELECT CUSTOMERS UP TO APPROXIMATE GOAL
    # =======================
    df_plan = df_plan.sort_values("Score_Cobro", ascending=False).reset_index(drop=True)
    df_plan["Monto_Acumulado"] = df_plan["Balance"].cumsum()

    df_seleccion = pd.DataFrame(columns=df_plan.columns)
    acumulado = 0.0

    for i, row in df_plan.iterrows():
        if acumulado >= meta_semanal:
            break
        df_seleccion = pd.concat([df_seleccion, pd.DataFrame([row])], ignore_index=True)
        acumulado += row["Balance"]

    # =======================
    # COMPLIANCE STATUS
    # =======================
    if es_futura:
        df_seleccion["Monto_Pagado"] = 0
        df_seleccion["Estado_Cumplimiento"] = "📅 Planned"
        monto_cobrado_semana = 0
        cumplimiento_semana = 0
    else:
        pagos_semana = comportamiento_pago[
            (comportamiento_pago["Fecha_fatura"] >= fecha_ini_semana) &
            (comportamiento_pago["Fecha_fatura"] <= fecha_fin_semana)
        ].groupby("Codigo")["Pagado"].sum().reset_index()

        df_seleccion = df_seleccion.merge(pagos_semana, on="Codigo", how="left").rename(columns={"Pagado": "Monto_Pagado"})
        df_seleccion["Monto_Pagado"] = df_seleccion["Monto_Pagado"].fillna(0)
        df_seleccion["Estado_Cumplimiento"] = df_seleccion["Monto_Pagado"].apply(lambda x: "Paid" if x > 0 else "Planned - Not fulfilled")

        monto_cobrado_semana = pagos_semana["Pagado"].sum()
        cumplimiento_semana = monto_cobrado_semana / meta_semanal if meta_semanal > 0 else 0

    # =======================
    # WEEKLY KPI
    # =======================
    monto_planificado = df_seleccion["Balance"].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Weekly goal", f"${meta_semanal:,.2f}")
    col2.metric("📋 Planned", f"${monto_planificado:,.2f}")
    col3.metric("✅ Collected", f"${monto_cobrado_semana:,.2f}", f"{cumplimiento_semana:.1%}" if not es_futura else "📅 Planned")

    # =======================
    # MONTHLY KPI
    # =======================
    mes_actual = fecha_ini_semana.month
    año_actual = fecha_ini_semana.year
    monto_cobrado_mes = comportamiento_pago[
        (comportamiento_pago["Fecha_fatura"].dt.month == mes_actual) &
        (comportamiento_pago["Fecha_fatura"].dt.year == año_actual)
    ]["Pagado"].sum()
    meta_mensual = meta_semanal * 4
    cumplimiento_mes = monto_cobrado_mes / meta_mensual if meta_mensual > 0 else 0
    st.markdown("### 📊 Global Monthly Indicator")
    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("🎯 Monthly goal", f"${meta_mensual:,.2f}")
    colm2.metric("💰 Monthly collected", f"${monto_cobrado_mes:,.2f}")
    colm3.metric("📈 Monthly compliance", f"{cumplimiento_mes:.1%}")

    # =======================
    # PLANNED TABLE
    # =======================
    st.subheader("📋 Customers planned for collection")
    st.dataframe(
        df_seleccion.assign(
            Balance=lambda x: x["Balance"].apply(lambda v: f"${v:,.2f}"),
            Probabilidad_Morosidad=lambda x: x["Probabilidad_Morosidad"].apply(lambda v: f"{v:.1%}"),
            HistorialPago_Normalizado=lambda x: x["HistorialPago_Normalizado"].apply(lambda v: f"{v:.2f}"),
            Balance_Normalizado=lambda x: x["Balance_Normalizado"].apply(lambda v: f"{v:.2f}"),
            Score_Cobro=lambda x: x["Score_Cobro"].apply(lambda v: f"{v:.2f}"),
            Monto_Pagado=lambda x: x["Monto_Pagado"].apply(lambda v: f"${v:,.2f}")
        ),
        hide_index=True,
        use_container_width=True
    )

    # =======================
    # CUSTOMERS OUTSIDE PLANNING WHO PAID
    # =======================
    if not es_futura:
        clientes_planificados = set(df_seleccion["Codigo"].unique())
        pagos_semana = comportamiento_pago[
            (comportamiento_pago["Fecha_fatura"] >= fecha_ini_semana) &
            (comportamiento_pago["Fecha_fatura"] <= fecha_fin_semana)
        ]
        pagos_fuera_plan = pagos_semana[~pagos_semana["Codigo"].isin(clientes_planificados)].copy()

        if not pagos_fuera_plan.empty:
            pagos_fuera_plan = pagos_fuera_plan.groupby("Codigo")["Pagado"].sum().reset_index()
            pagos_fuera_plan = pagos_fuera_plan.merge(
                estado_cuenta[['Codigo', 'Nombre Cliente', 'Probabilidad_Morosidad']].drop_duplicates(),
                on='Codigo',
                how='left'
            )
            st.subheader("💵 Customers outside planning who paid")
            st.dataframe(
                pagos_fuera_plan.assign(
                    Pagado=lambda x: x["Pagado"].apply(lambda v: f"${v:,.2f}"),
                    Probabilidad_Morosidad=lambda x: x["Probabilidad_Morosidad"].apply(lambda v: f"{v:.1%}")
                ).rename(columns={"Pagado": "Amount Paid", "Probabilidad_Morosidad": "Risk"}),
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No payments outside planning this week.")

    # =======================
    # PLANNING EXPORT
    # =======================
    csv_export = df_seleccion.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download weekly plan (CSV)",
        data=csv_export,
        file_name=f"collection_plan_{semana_sel.replace(' ', '_')}.csv",
        mime="text/csv"
    )

    # =======================
    # METHOD EXPLANATION
    # =======================
    st.markdown("""
    ---
    **🧮 Prioritization method used**  
    Customers are prioritized based on a **collection score** calculated as:

    **Score = (0.5 × (1 - Delinquency Probability)) + (0.3 × Normalized Payment History) + (0.2 × Normalized Balance)**  

    - **Delinquency Probability**: estimated by credit risk model.
    - **Normalized Payment History**: average payment history compared to the highest among all customers.
    - **Normalized Balance**: pending balance compared to maximum balance in portfolio.

    Each customer's status is determined by whether they **paid** in the selected week or were **planned - not fulfilled**.
    """)

# =============================================
# TAB 8: CREDIT LIMIT BY CUSTOMER
# =============================================

with tab8:
    st.header("💳 Recommended Credit Limits", divider="blue")
    
    # Function to calculate limits (same as provided before)
    def calcular_limites_credito(estado_cuenta, comportamiento_pago):
        """
        Calculates recommended credit limits for each customer based on:
        - Payment history
        - Delinquency probability
        - Risk segment
        - Current behavior
        """
        try:
            # Group data by customer
            clientes_agg = estado_cuenta.groupby(['Codigo', 'Nombre Cliente']).agg({
                'Balance': 'sum',
                'Dias': 'mean',
                'Probabilidad_Morosidad': 'mean',
                'Segmento_Riesgo': lambda x: x.value_counts().index[0],
                'Estado_Morosidad': lambda x: x.value_counts().index[0]
            }).reset_index()

            # Get payment history for each customer
            if not comportamiento_pago.empty:
                historial_pagos = comportamiento_pago.groupby('Codigo').agg({
                    'Pagado': ['sum', 'mean', 'count'],
                    'Fecha_fatura': ['min', 'max']
                }).reset_index()
                
                # Flatten multiindex columns
                historial_pagos.columns = ['_'.join(col).strip() for col in historial_pagos.columns.values]
                historial_pagos = historial_pagos.rename(columns={
                    'Codigo_': 'Codigo',
                    'Pagado_sum': 'Total_Pagado',
                    'Pagado_mean': 'Promedio_Pago',
                    'Pagado_count': 'Cantidad_Pagos',
                    'Fecha_fatura_min': 'Primer_Pago',
                    'Fecha_fatura_max': 'Ultimo_Pago'
                })
                
                # Calculate customer seniority (in days)
                historial_pagos['Antiguedad'] = (pd.to_datetime('today') - historial_pagos['Ultimo_Pago']).dt.days
                
                # Join with main data
                clientes_agg = clientes_agg.merge(historial_pagos, on='Codigo', how='left')
            else:
                # If no payment data, create empty columns
                clientes_agg['Total_Pagado'] = 0
                clientes_agg['Promedio_Pago'] = 0
                clientes_agg['Cantidad_Pagos'] = 0
                clientes_agg['Antiguedad'] = 0

            # =============================================
            # CREDIT LIMIT FACTOR CALCULATION
            # =============================================
            
            # 1. Risk factor (inversely proportional to delinquency probability)
            clientes_agg['Factor_Riesgo'] = 1 - clientes_agg['Probabilidad_Morosidad']
            
            # 2. Payment history factor (based on quantity and consistency of payments)
            if not comportamiento_pago.empty:
                max_pagos = clientes_agg['Cantidad_Pagos'].max()
                clientes_agg['Factor_Historial'] = (
                    0.5 * (clientes_agg['Cantidad_Pagos'] / max_pagos) +
                    0.5 * (1 - (clientes_agg['Promedio_Pago'].std() / clientes_agg['Promedio_Pago'].mean()))
                ).clip(0, 1)
            else:
                clientes_agg['Factor_Historial'] = 0.5  # Neutral value if no data
            
            # 3. Seniority factor (older customers get more trust)
            if 'Antiguedad' in clientes_agg.columns:
                max_antiguedad = clientes_agg['Antiguedad'].max() or 1
                clientes_agg['Factor_Antiguedad'] = (clientes_agg['Antiguedad'] / max_antiguedad).clip(0, 1)
            else:
                clientes_agg['Factor_Antiguedad'] = 0.5
            
            # 4. Current behavior factor (based on days overdue)
            clientes_agg['Factor_Comportamiento'] = np.where(
                clientes_agg['Dias'] > 90, 0.1,
                np.where(
                    clientes_agg['Dias'] > 60, 0.3,
                    np.where(
                        clientes_agg['Dias'] > 30, 0.6,
                        np.where(
                            clientes_agg['Dias'] > 15, 0.8,
                            1.0
                        )
                    )
                )
            )
            
            # =============================================
            # LIMIT CALCULATION
            # =============================================
            
            # Base amount for calculation (payment average or current balance)
            clientes_agg['Monto_Base'] = np.where(
                clientes_agg['Promedio_Pago'] > 0,
                clientes_agg['Promedio_Pago'],
                clientes_agg['Balance']
            )
            
            # Factor weighting
            clientes_agg['Ponderacion'] = (
                0.4 * clientes_agg['Factor_Riesgo'] +
                0.3 * clientes_agg['Factor_Historial'] +
                0.2 * clientes_agg['Factor_Antiguedad'] +
                0.1 * clientes_agg['Factor_Comportamiento']
            )
            
            # Recommended credit limit (adjusted by weighting)
            clientes_agg['Limite_Credito'] = clientes_agg['Monto_Base'] * (1 + clientes_agg['Ponderacion'])
            
            # Adjust limits for high-risk customers
            clientes_agg['Limite_Credito'] = np.where(
                clientes_agg['Segmento_Riesgo'].isin(['High (60-80%)', 'Extreme (80-100%)']),
                clientes_agg['Limite_Credito'] * 0.5,  # Reduce by half for high risk
                clientes_agg['Limite_Credito']
            )
            
            # Minimum limit of $1,000 for all customers
            clientes_agg['Limite_Credito'] = clientes_agg['Limite_Credito'].clip(lower=1000)
            
            # =============================================
            # MAXIMUM CREDIT DAYS CALCULATION
            # =============================================
            
            clientes_agg['Dias_Maximos'] = np.where(
                clientes_agg['Segmento_Riesgo'] == 'Extreme (80-100%)', 7,
                np.where(
                    clientes_agg['Segmento_Riesgo'] == 'High (60-80%)', 15,
                    np.where(
                        clientes_agg['Segmento_Riesgo'] == 'Moderate (30-60%)', 30,
                        60  # For low risk
                    )
                )
            )
            
            # Adjust days based on recent behavior
            clientes_agg['Dias_Maximos'] = np.where(
                clientes_agg['Dias'] > 60,
                clientes_agg['Dias_Maximos'] * 0.5,  # Reduce by half if high overdue
                clientes_agg['Dias_Maximos']
            ).astype(int)
            
            # =============================================
            # ASSIGNMENT EXPLANATION
            # =============================================
            
            def generar_explicacion(row):
                explicaciones = {
                    'Factor_Riesgo': f"Delinquency risk of {row['Probabilidad_Morosidad']:.1%}",
                    'Factor_Historial': f"History of {row['Cantidad_Pagos']} payments with average ${row['Promedio_Pago']:,.2f}" if row['Cantidad_Pagos'] > 0 else "No payment history",
                    'Factor_Antiguedad': f"Seniority of {row['Antiguedad']} days as customer" if row['Antiguedad'] > 0 else "New customer",
                    'Factor_Comportamiento': f"Current behavior: {row['Estado_Morosidad']} (average {row['Dias']:.0f} days overdue)"
                }
                
                segmento = {
                    'Low (0-30%)': "low risk",
                    'Moderate (30-60%)': "moderate risk",
                    'High (60-80%)': "high risk",
                    'Extreme (80-100%)': "extreme risk"
                }.get(row['Segmento_Riesgo'], "unknown risk")
                
                return (
                    f"Customer classified as {segmento}. " +
                    f"Based on: {explicaciones['Factor_Riesgo']}, {explicaciones['Factor_Historial']}, " +
                    f"{explicaciones['Factor_Antiguedad']}, {explicaciones['Factor_Comportamiento']}. " +
                    f"Recommendation: {'reduce' if row['Ponderacion'] < 0.5 else 'maintain/increase'} limit."
                )
            
            clientes_agg['Explicacion'] = clientes_agg.apply(generar_explicacion, axis=1)
            
            # Select and order final columns
            columnas_finales = [
                'Codigo', 'Nombre Cliente', 'Segmento_Riesgo', 'Estado_Morosidad',
                'Probabilidad_Morosidad', 'Balance', 'Total_Pagado', 'Cantidad_Pagos',
                'Limite_Credito', 'Dias_Maximos', 'Explicacion'
            ]
            
            return clientes_agg[columnas_finales].sort_values('Limite_Credito', ascending=False)
        
        except Exception as e:
            st.error(f"Error calculating credit limits: {str(e)}")
            return pd.DataFrame()

    # Execute calculation
    limites_clientes = calcular_limites_credito(estado_cuenta, comportamiento_pago)

    if not limites_clientes.empty:
        # Filters for table
        st.subheader("🔍 Filters", divider="gray")
        
        col1, col2 = st.columns(2)
        
        with col1:
            segmentos = st.multiselect(
                "Filter by risk segment",
                options=limites_clientes['Segmento_Riesgo'].unique(),
                default=limites_clientes['Segmento_Riesgo'].unique()
            )
            
        with col2:
            min_limite, max_limite = st.slider(
                "Credit limit range",
                min_value=float(limites_clientes['Limite_Credito'].min()),
                max_value=float(limites_clientes['Limite_Credito'].max()),
                value=(float(limites_clientes['Limite_Credito'].min()), float(limites_clientes['Limite_Credito'].max()))
            )
        
        # Apply filters
        df_filtrado = limites_clientes[
            (limites_clientes['Segmento_Riesgo'].isin(segmentos)) &
            (limites_clientes['Limite_Credito'] >= min_limite) &
            (limites_clientes['Limite_Credito'] <= max_limite)
        ]
        
        # Show summary metrics
        st.subheader("📊 Limits Summary", divider="gray")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Customers", len(df_filtrado))
        col2.metric("Average Limit", f"${df_filtrado['Limite_Credito'].mean():,.2f}")
        col3.metric("Average Days", f"{df_filtrado['Dias_Maximos'].mean():.0f} days")
        
        # Show table with limits
        st.subheader("📋 Recommended Limits by Customer", divider="gray")
        
        # Format columns for better visualization
        df_mostrar = df_filtrado.assign(
            Balance=lambda x: x['Balance'].apply(lambda v: f"${v:,.2f}"),
            Total_Pagado=lambda x: x['Total_Pagado'].apply(lambda v: f"${v:,.2f}"),
            Probabilidad_Morosidad=lambda x: x['Probabilidad_Morosidad'].apply(lambda v: f"{v:.1%}"),
            Limite_Credito=lambda x: x['Limite_Credito'].apply(lambda v: f"${v:,.2f}")
        )
        
        st.dataframe(
            df_mostrar,
            column_config={
                "Explicacion": st.column_config.TextColumn(
                    "Explanation",
                    help="Detail of how the limit was calculated for this customer",
                    width="large"
                )
            },
            hide_index=True,
            use_container_width=True,
            height=600
        )
        
        # Limit distribution chart
        st.subheader("📈 Limit Distribution", divider="gray")
        
        fig = px.histogram(
            df_filtrado,
            x='Limite_Credito',
            nbins=20,
            title='Recommended Credit Limits Distribution',
            labels={'Limite_Credito': 'Credit Limit ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Days vs limit chart
        fig = px.scatter(
            df_filtrado,
            x='Dias_Maximos',
            y='Limite_Credito',
            color='Segmento_Riesgo',
            hover_data=['Nombre Cliente'],
            title='Relationship between Credit Days and Limit Amount',
            labels={
                'Dias_Maximos': 'Maximum Credit Days',
                'Limite_Credito': 'Credit Limit ($)'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Download option
        st.subheader("📤 Export Results", divider="gray")
        
        csv = df_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download credit limits (CSV)",
            data=csv,
            file_name="customer_credit_limits.csv",
            mime="text/csv"
        )
        
        # Methodological explanation
        with st.expander("📝 View detailed methodology"):
            st.markdown("""
            ### Limit Calculation Methodology
            
            Credit limits were calculated considering the following factors:
            
            1. **Risk Factor (40% weight)**:
               - Based on delinquency probability from predictive model
               - Formula: `1 - Delinquency_Probability`
            
            2. **Payment History Factor (30% weight)**:
               - Considers payment quantity and consistency (standard deviation)
               - Formula: `0.5*(n_payments/max_payments) + 0.5*(1 - std_payments/mean_payments)`
            
            3. **Seniority Factor (20% weight)**:
               - Older customers receive more trust
               - Formula: `days_seniority / max_seniority`
            
            4. **Behavior Factor (10% weight)**:
               - Based on current days overdue
               - Scale from 0.1 (very delinquent) to 1.0 (current)
            
            **Limit Calculation**:
            ```
            Base_Amount = MAX(Average_Payments, Current_Balance)
            Weighting = 0.4*Risk_Factor + 0.3*History_Factor + 0.2*Seniority_Factor + 0.1*Behavior_Factor
            Credit_Limit = Base_Amount * (1 + Weighting)
            ```
            
            **Special Adjustments**:
            - High-risk customers (60-100% probability) receive 50% of calculated limit
            - Minimum guaranteed limit of $1,000 for all customers
            - Maximum days reduced by half for customers with >60 days overdue
            """)
    else:
        st.warning("Could not calculate credit limits. Please check available data.")

# =============================================
# GLOBAL FILTERS (sidebar)
# =============================================
with st.sidebar:
    st.title("⚙️ Filters")
    
    min_date = estado_cuenta['Fecha_fatura'].min().to_pydatetime()
    max_date = estado_cuenta['Fecha_fatura'].max().to_pydatetime()
    
    fecha_inicio = st.date_input(
        "Start date",
        min_date,
        min_value=min_date,
        max_value=max_date
    )
    
    fecha_fin = st.date_input(
        "End date",
        max_date,
        min_value=min_date,
        max_value=max_date
    )
    
    estados = st.multiselect(
        "Delinquency statuses",
        options=estado_cuenta['Estado_Morosidad'].unique(),
        default=estado_cuenta['Estado_Morosidad'].unique()
    )
    
    if st.button("Apply Filters"):
        estado_cuenta = estado_cuenta[
            (estado_cuenta['Fecha_fatura'] >= pd.to_datetime(fecha_inicio)) & 
            (estado_cuenta['Fecha_fatura'] <= pd.to_datetime(fecha_fin)) &
            (estado_cuenta['Estado_Morosidad'].isin(estados))
        ]
        st.rerun()

# =============================================
# DATA EXPORT
# =============================================
with st.sidebar:
    st.title("📤 Export Data")
    
    if st.download_button(
        label="Download Data (CSV)",
        data=estado_cuenta.to_csv(index=False).encode('utf-8'),
        file_name="delinquency_report.csv",
        mime="text/csv"
    ):
        st.success("Data exported successfully")

# =============================================
# FOOTER
# =============================================
st.sidebar.markdown("---")
st.sidebar.info("""
    **© 2004 Erick Geronimo. All rights reserved**  
    Version 2.0 - June 2024
""")
