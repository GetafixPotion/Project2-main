import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load and preprocess data
df = pd.read_csv('churn-bigml-80.csv')
df.drop(columns=['Area code', 'State'], inplace=True)

# Combine redundant columns
df['Total Calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
df['Total Minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
df['Total Charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']

df.drop(columns=[
    'Number vmail messages', 'Total eve calls', 'Total eve charge', 'Total eve minutes',
    'Total day calls', 'Total day charge', 'Total day minutes', 'Total night calls',
    'Total night charge', 'Total night minutes', 'Total intl charge', 'Total intl minutes'
], inplace=True)

# Encode categorical variables
df['Churn'] = df['Churn'].map({True: 1, False: 0})
df['International plan'] = df['International plan'].map({'Yes': 1, 'No': 0})
df['Voice mail plan'] = df['Voice mail plan'].map({'Yes': 1, 'No': 0})

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
app.layout = dbc.Container([
    html.H2("Customer Churn Prediction", className="my-4 text-center"),

    dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("Account Length", html_for="account_length"),
                dbc.Input(id='account_length', type='number', value=100),

                dbc.Label("International Plan (1 = Yes, 0 = No)", html_for="intl_plan"),
                dbc.Input(id='intl_plan', type='number', min=0, max=1, value=0),

                dbc.Label("Voice Mail Plan (1 = Yes, 0 = No)", html_for="vmail_plan"),
                dbc.Input(id='vmail_plan', type='number', min=0, max=1, value=0),

                dbc.Label("Total Intl Calls", html_for="intl_calls"),
                dbc.Input(id='intl_calls', type='number', value=4),
            ], md=6),

            dbc.Col([
                dbc.Label("Customer Service Calls", html_for="cust_serv_calls"),
                dbc.Input(id='cust_serv_calls', type='number', value=1),

                dbc.Label("Total Calls", html_for="total_calls"),
                dbc.Input(id='total_calls', type='number', value=100),

                dbc.Label("Total Minutes", html_for="total_minutes"),
                dbc.Input(id='total_minutes', type='number', value=200.0),

                dbc.Label("Total Charge", html_for="total_charge"),
                dbc.Input(id='total_charge', type='number', value=30.0),
            ], md=6),
        ]),

        dbc.Button("Predict Churn", id="predict-btn", color="primary", className="my-3")
        
]),

    html.Hr(),
    html.Div(id='prediction-result', className="text-center fs-4 fw-bold text-info"),
])


# Callback to handle prediction
@app.callback(
    Output('prediction-result', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('account_length', 'value'),
    State('intl_plan', 'value'),
    State('vmail_plan', 'value'),
    State('intl_calls', 'value'),
    State('cust_serv_calls', 'value'),
    State('total_calls', 'value'),
    State('total_minutes', 'value'),
    State('total_charge', 'value'),
)
def predict_churn(n_clicks, al, ip, vmp, tic, csc, tc, tm, tcharge):
    if n_clicks:
        input_data = np.array([[al, ip, vmp, tic, csc, tc, tm, tcharge]])
        input_scaled = scaler.transform(input_data)
        prediction = mlp_model.predict(input_scaled)[0][0]
        churn_label = "Likely to Churn" if prediction >= 0.5 else "Not Likely to Churn"
        return f"Churn Probability: {prediction:.2f} — {churn_label}"
    return ""

# Run server
if __name__ == '__main__':
    app.run(debug=True)
