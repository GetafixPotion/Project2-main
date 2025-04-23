import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import tensorflow
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import plotly.express as px
import os

# Path to CSV files
csv_file_path = "churn-bigml-80.csv"
Procesed_data_csv = 'ProcessedData.csv'

# Load and preprocess data
df = pd.read_csv(csv_file_path)
df.drop(columns=['Area code', 'State'], inplace=True)

#combine redundant columns and drop them
df['Total Calls'] = (df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls'])
df['Total Minutes'] = ( df['Total day minutes'] +df['Total eve minutes'] +df['Total night minutes']+df['Total intl minutes'])
df['Total Charge'] = (df['Total day charge'] +df['Total eve charge'] + df['Total night charge']+df['Total intl charge'])
df['Plan'] = ((df['International plan'] == 'Yes') | (df['Voice mail plan'] == 'Yes')).astype(bool)
df.drop(columns=['Voice mail plan','International plan','Number vmail messages','Total eve calls','Total eve charge','Total eve minutes',
                 'Total day calls','Total day charge','Total day minutes','Total night calls','Total night charge','Total night minutes',
                 'Total intl charge','Total intl minutes','Total intl calls'], inplace= True)

churn_counts = df['Churn'].value_counts()
churn_bar_chart = px.bar(
    x=churn_counts.index,
    y=churn_counts.values,
    labels={'x': 'Churn Status', 'y': 'Number of Customers'},
    title="Churn vs. Non-Churned Customers"
)
# Additional statistics
total_customers = len(df)
churned_customers = churn_counts[True] if True in churn_counts.index else 0
non_churned_customers = churn_counts[False] if False in churn_counts.index else 0
churn_percentage = (churned_customers / total_customers) * 100
avg_account_length = df['Account length'].mean()
avg_customer_service_calls = df['Customer service calls'].mean()
# Encode categorical variables
df['Churn'] = df['Churn'].map({True: 1, False: 0})


# Stats layout
stats_layout = html.Div([
    html.H3("Customer Statistics"),
    html.P(f"Total number of customers: {total_customers}"),
    html.P(f"Number of customers who churned: {churned_customers}"),
    html.P(f"Number of customers who did not churn: {non_churned_customers}"),
    html.P(f"Churn rate: {churn_percentage:.2f}%"),
    html.P(f"Average account length: {avg_account_length:.2f} months"),
    html.P(f"Average customer service calls per customer: {avg_customer_service_calls:.2f}")
])



X = df.drop("Churn", axis=1)
y = df["Churn"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build the MLP model
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)

# Initialize Dash app with Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Churn Prediction App"

# Layout with cards and responsive form
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the app layout
app.layout = dbc.Container([
    html.H2("Customer Churn Prediction", className="my-4 text-center"),

    dbc.Row([
        dbc.Col([
            html.H4("Load and Display Data", className="mb-3"),
            html.Button("Load Dataset", id='load-data-btn', n_clicks=0, className="btn btn-primary w-100 mb-2"),
            html.Div(id='output-data-load', style={'marginTop': '10px'}),
            html.Iframe(id='csv-iframe', srcDoc='', style={'width': '100%', 'height': '400px', 'display': 'none'}),
            html.Hr(),
            html.Button("Load Processed Data", id='load-processed-btn', n_clicks=0, className="btn btn-secondary w-100 mb-2"),
            html.Div(id='output-processed-load', style={'marginTop': '10px'}),
            html.Iframe(id='processed-iframe', srcDoc='', style={'width': '100%', 'height': '400px', 'display': 'none'}),
        ], md=6),

        dbc.Col([
            dbc.Label("Account Length", html_for="account_length"),
            dbc.Input(id='account_length', type='number'),

            dbc.Label("Plan"),
            dcc.Dropdown(
                id='plan',
                options=[{'label': 'Yes', 'value': 1}, {'label': 'No', 'value': 0}],
                placeholder='Select...',
                className="form-control"
            ),
        ], md=6),
    

    dbc.Row([
        dbc.Col([
            dbc.Label("Customer Service Calls", html_for="cust_serv_calls"),
            dbc.Input(id='cust_serv_calls', type='number'),
        ], md=6),

        dbc.Col([
            dbc.Label("Total Calls", html_for="total_calls"),
            dbc.Input(id='total_calls', type='number'),
        ], md=6),

        dbc.Col([
            dbc.Label("Total Minutes", html_for="total_minutes"),
            dbc.Input(id='total_minutes', type='number'),
        ], md=6),

        dbc.Col([
            dbc.Label("Total Charge", html_for="total_charge"),
            dbc.Input(id='total_charge', type='number'),
        ], md=6),
    ]),]),

    dbc.Row([
        dbc.Col([
            dbc.Button("Predict Churn", id="predict-btn", color="primary", className="my-3"),
        ], md=12),
    ]),

    html.Hr(),
    html.Div([
        html.H4("Prediction Result", className="text-center"),
        html.Div(id='prediction-result', className="text-center fs-4 fw-bold text-info", style={'border': '2px solid #ccc', 'padding': '10px', 'margin': '20px auto', 'max-width': '600px'})
    ]),
    
])

@app.callback(
    Output('csv-iframe', 'srcDoc'),
    Output('csv-iframe', 'style'),
    Input('load-data-btn', 'n_clicks')
)
def load_csv_in_iframe(n_clicks):
    if n_clicks > 0:
        # Read the CSV and convert to HTML table
        try:
            df = pd.read_csv(csv_file_path)
            html_table = df.to_html(classes='table table-striped', index=False)  # Convert to HTML

            # Display the iframe with the HTML table content
            return html_table, {'width': '100%', 'height': '400px', 'display': 'block'}
        except Exception as e:
            return f'Error loading CSV: {e}', {'display': 'none'}
    return '', {'display': 'none'}

@app.callback(
    Output('processed-iframe', 'srcDoc'),
    Output('processed-iframe', 'style'),
    Input('load-processed-btn', 'n_clicks'),
)
def load_processed_csv(n_clicks):
    if n_clicks > 0:
        try:
            df = pd.read_csv(Procesed_data_csv)
            html_table = df.to_html(classes='table table-striped', index=False)
            return html_table, {'width': '100%', 'height': '400px', 'display': 'block'}
        except Exception as e:
            return f'Error loading CSV: {e}', {'display': 'none'}
    return '', {'display': 'none'}


# Callback to handle prediction
@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('account_length', 'value'),
    State('cust_serv_calls', 'value'),
    State('total_calls', 'value'),
    State('total_minutes', 'value'),
    State('total_charge', 'value'),
    State('plan', 'value')
)
def predict_churn(n_clicks, al, ip, tc, tm, tcharge, csc):
    # Ensure inputs are not None or empty
    if n_clicks is None or n_clicks == 0:
        return ""  # No prediction if button hasn't been clicked yet
    
    if None in [al, ip, tc, tm, tcharge, csc] or '' in [al, ip, tc, tm, tcharge, csc]:
        return "Error: Please fill in all input fields."

    try:
        # Reshape input data for prediction
        input_data = np.array([[al, ip, tc, tm, tcharge, csc]])

        # Ensure input data is scaled correctly
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = mlp_model.predict(input_scaled)[0][0]

        # Determine churn likelihood
        churn_label = "Likely to Churn" if prediction >= 0.5 else "Not Likely to Churn"

        return f"Churn Probability: {prediction:.2f} — {churn_label}"
    except Exception as e:
        return f"Error during prediction: {str(e)}"
    
port = int(os.environ.get("PORT", 10000))

# Run server
if __name__ == '__main__':
    app.run(debug=True)
    app.run_server(host="0.0.0.0", port=port)
