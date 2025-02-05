import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import joblib
import numpy as np
import plotly.graph_objects as go  # For the gauge
import plotly.express as px  # For the bar plot, scatter plot, and heatmap
import pandas as pd  # For the DataFrame
import dash_bootstrap_components as dbc  # For Bootstrap components
import io  # For handling file downloads

# Load the trained model
model = joblib.load("recovery_time_model.pkl")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Function to create the gauge
def create_gauge(prediction):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={"text": "Predicted Recovery Time"},
        gauge={"axis": {"range": [0, 21]}, "bar": {"color": "green"}}
    ))
    return fig

# Data for the bar plot
df = pd.DataFrame({
    "Drug Administered": ["Ceftriaxon", "Cristaline Pencillin", "Ampicillin & Gentamicin"],
    "Avg Recovery Time": [6.5, 7.2, 5.8]
})

# Create the bar plot
bar_fig = px.bar(df, x="Drug Administered", y="Avg Recovery Time", title="Average Recovery Time per Drug", 
                      color="Drug Administered", color_discrete_sequence=["#a83252", "#4a32a8", "#a89332"])

# Legend data
feature_descriptions = [
    {"Feature": "Age (Months)", "Description": "The age of the patient in months."},
    {"Feature": "Hospitalization Duration (Days)", "Description": "The number of days the patient has been hospitalized."},
    {"Feature": "Signs of Symptoms", "Description": "Indicates whether the patient shows signs of symptoms (Yes/No)."},
    {"Feature": "Temperature (°C)", "Description": "The body temperature of the patient in degrees Celsius."},
    {"Feature": "Respiratory Rate (Breaths/Min)", "Description": "The number of breaths per minute for the patient."},
    {"Feature": "Kaplan Meier Survival Estimate", "Description": "A statistical estimate of survival probability."},
    {"Feature": "Hazard Function", "Description": "A measure of the risk of an event occurring at a given time."},
    {"Feature": "Time Factor", "Description": "A factor representing the time since the onset of the condition."},
    {"Feature": "Clinical Score", "Description": "A score representing the clinical severity of the condition, higher means critical."},
    {"Feature": "Drug Administered", "Description": "The type of drug administered to the patient."},
]

# Convert legend data to a DataFrame for display
legend_df = pd.DataFrame(feature_descriptions)

# Product description text
product_description = """
This tool is designed to predict the estimated recovery time for pneumonia patients based on clinical inputs. 
It uses an advanced machine learning algorithm to analyze various factors such as age, hospitalization duration, 
symptoms, and drug administration to provide an accurate recovery time prediction.

### How It Works:
1. **Input Clinical Data**: Enter the patient's clinical data, such as age, temperature, respiratory rate, and more.
2. **Predict Recovery Time**: Click the "Predict" button to generate the estimated recovery time.
3. **Visualize Results**: View the predicted recovery time in text, a gauge chart, and an animated progress bar.

### How to Use:
- Fill in all the required fields with accurate clinical data.
- Click the "Predict" button to get the recovery time.
- Use the visualizations (gauge and progress bar) to track the recovery progress.
- Refer to the "Feature Descriptions" section for explanations of each input field.
- After receiving the prediction, users can click the download button to save their information.
"""

# Model accuracy metrics
model_metrics = {
    'Mean Absolute Error (MAE)': 0.01446,
    'Mean Squared Error (MSE)': 0.00051,
    'Root Mean Squared Error (RMSE)': 0.02236,
    'R-Squared (R²)': 0.96861
}

# Convert metrics to DataFrame
metrics_df = pd.DataFrame({
    "Metric": list(model_metrics.keys()),
    "Value": list(model_metrics.values())
})

# Accuracy explanations
accuracy_explanation = """
### Model Accuracy Metrics Explained:

- **Mean Absolute Error (MAE)**: 0.0145  
  The average absolute difference between predicted and actual values. Lower is better.

- **Mean Squared Error (MSE)**: 0.00050  
  The average squared difference between predictions and actuals. Punishes larger errors more.

- **Root Mean Squared Error (RMSE)**: 0.0224  
  The square root of MSE, in the same units as the target variable.

- **R-Squared (R²)**: 0.9686  
  Proportion of variance explained by the model (0-1). Closer to 1 is better.

**Interpretation**:  
With an R² of 96.86% and extremely low error metrics, the model demonstrates exceptional predictive accuracy.
"""

app.layout = html.Div(
    style={'backgroundImage': 'url("/assets/blue.jpg")', 
           'backgroundSize': 'cover',
           'backgroundPosition': 'center',
           'minHeight': '100vh',
           'padding': '30px'},
    children=[
        html.H1("Pneumonia Recovery Time Predictive Tool 🩺", style={'textAlign': 'center', 'fontWeight': 'bold'}),
        html.H4("By Eli Kekombe™", style={'textAlign': 'center'}),
        html.P("This tool predicts the estimated recovery time for pneumonia patients based on clinical inputs. The tool uses an advanced machine learning algorithm to estimate recovery time.",
               style={'textAlign': 'center'}),

        # Collapsible sections buttons
        html.Div([
            dbc.Button(
                "Model Accuracy",
                id="metrics-collapse-button",
                className="mb-3",
                color="primary",
                style={'width': '220px', 'margin': '10px'}
            ),
            dbc.Button(
                "Show Feature Descriptions",
                id="legend-collapse-button",
                className="mb-3",
                color="primary",
                style={'width': '220px', 'margin': '10px'}
            ),
            dbc.Button(
                "Product Description",
                id="product-collapse-button",
                className="mb-3",
                color="primary",
                style={'width': '220px', 'margin': '10px'}
            ),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexWrap': 'wrap'}),

        # Model metrics collapse
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody([
                    html.H4("Model Performance Metrics", style={'fontWeight': 'bold'}),
                    dbc.Table.from_dataframe(
                        metrics_df,
                        striped=True,
                        bordered=True,
                        hover=True,
                        style={'width': '60%', 'margin': '20px auto'}
                    ),
                    dcc.Markdown(accuracy_explanation)
                ]),
                style={'width': '80%', 'margin': '20px auto'}
            ),
            id="metrics-collapse",
            is_open=False,
        ),

        # Feature descriptions collapse
        dbc.Collapse(
            dbc.Table.from_dataframe(
                legend_df,
                striped=True,
                bordered=True,
                hover=True,
                style={'width': '80%', 'margin': '20px auto', 'backgroundColor': 'white'}
            ),
            id="legend-collapse",
            is_open=False,
        ),

        # Product description collapse
        dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    html.Div([
                        html.H4("Product Description", style={'fontWeight': 'bold'}),
                        dcc.Markdown(product_description)
                    ])
                ),
                style={'width': '80%', 'margin': '20px auto'}
            ),
            id="product-collapse",
            is_open=False,
        ),

        # Input fields
        html.Div([
            html.Div([
                html.Label("Age (Months)", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='age', type='number', min=0, max=1200, value=6, step=1, style={'width': '120px', 'margin': '0 auto', 'display': 'block'})
            ], style={'flex': '1', 'margin': '10px'}),
            html.Div([
                html.Label("Hospitalization Duration (Days)", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='duration', type='number', min=1, max=30, value=7, step=1, style={'width': '120px', 'margin': '0 auto', 'display': 'block'})
            ], style={'flex': '1', 'margin': '10px'}),
            html.Div([
                html.Label("Signs of Symptoms", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Dropdown(id='signs', options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ], value=1, style={'width': '150px', 'margin': '0 auto', 'display': 'block', 'backgroundColor': 'white'})
            ], style={'flex': '1', 'margin': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexWrap': 'wrap'}),

        html.Div([
            html.Div([
                html.Label("Temperature (°C)", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='temperature', type='number', min=35.0, max=42.0, value=37.0, step=0.1, style={'width': '120px', 'margin': '0 auto', 'display': 'block'})
            ], style={'flex': '1', 'margin': '10px'}),
            html.Div([
                html.Label("Respiratory Rate (Breaths/Min)", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='respiratory_rate', type='number', min=10, max=100, value=50, step=1, style={'width': '120px', 'margin': '0 auto', 'display': 'block'})
            ], style={'flex': '1', 'margin': '10px'}),
            html.Div([
                html.Label("Kaplan Meier Survival Estimate", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='km', type='number', min=0.0, max=1.0, value=0.5, step=0.01, style={'width': '120px', 'margin': '0 auto', 'display': 'block'})
            ], style={'flex': '1', 'margin': '10px'}),
            html.Div([
                html.Label("Hazard Function", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='hazard_function', type='number', min=0.0, max=5.0, value=1.0, step=0.1, style={'width': '120px', 'margin': '0 auto', 'display': 'block', 'backgroundColor': 'white'})
            ], style={'flex': '1', 'margin': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexWrap': 'wrap'}),

        html.Div([
            html.Div([
                html.Label("Time Factor", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='time', type='number', min=1, max=100, value=10, step=1, style={'width': '120px', 'margin': '0 auto', 'display': 'block'})
            ], style={'flex': '1', 'margin': '10px'}),
            html.Div([
                html.Label("Clinical Score", style={'display': 'block', 'textAlign': 'center'}),
                dcc.Input(id='clinical_score', type='number', min=0.0, max=18.0, value=5.0, step=0.1, style={'width': '120px', 'margin': '0 auto', 'display': 'block'})
            ], style={'flex': '1', 'margin': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'flexWrap': 'wrap'}),

        html.Div([
            html.Label("Drug Administered", style={'display': 'block', 'textAlign': 'center'}),
            dcc.Dropdown(id='drug_administered', options=[
                {'label': 'Ceftriaxon', 'value': 'ceftriaxon'},
                {'label': 'Cristaline Pencillin', 'value': 'cristaline_pencillin'},
                {'label': 'Ampicillin & Gentamicin', 'value': 'ampicillin_gentamicin'}
            ], value='ceftriaxon', style={'width': '250px', 'margin': '0 auto', 'display': 'block'})
        ], style={'textAlign': 'center', 'margin': '10px'}),

        html.Button('Predict', id='predict-button', n_clicks=0, style={'width': '220px', 'margin': '20px auto', 'display': 'block', 'fontWeight': 'bold', 'backgroundColor': '#007BFF', 'color': 'white', 'border': 'none', 'padding': '10px', 'borderRadius': '5px'}), 

        # Output for the prediction text
        html.H3(id='output-prediction', style={'textAlign': 'center', 'marginTop': '20px', 'fontWeight': 'bold'}),

        # Gauge visualization
        dcc.Graph(id='prediction-gauge', style={'margin': '20px auto', 'width': '50%'}),

        # Animated recovery progress bar
        dbc.Progress(id='recovery-progress', value=0, max=15, striped=True, animated=True, style={'width': '80%', 'margin': '20px auto'}),

        # Bar plot for average recovery time per drug
        dcc.Graph(id='drug-recovery-plot', figure=bar_fig, style={'margin': '20px auto', 'width': '80%'}),

        # Dropdown to select feature for scatter plot
        html.Div([
            html.Label("Select Feature for Scatter Plot", style={'display': 'block', 'textAlign': 'center'}),
            dcc.Dropdown(
                id='scatter-feature',
                options=[
                    {'label': 'Age (Months)', 'value': 'age'},
                    {'label': 'Hospitalization Duration (Days)', 'value': 'duration'},
                    {'label': 'Temperature (°C)', 'value': 'temperature'},
                    {'label': 'Respiratory Rate (Breaths/Min)', 'value': 'respiratory_rate'},
                    {'label': 'Time Factor', 'value': 'time'},
                    {'label': 'Clinical Score', 'value': 'clinical_score'},
                ],
                value='age',  # Default feature
                style={'width': '250px', 'margin': '0 auto', 'display': 'block'}
            ),
        ], style={'textAlign': 'center', 'margin': '10px'}),

        # Scatter plot for recovery time vs. selected feature
        dcc.Graph(id='recovery-scatter-plot', style={'margin': '20px auto', 'width': '80%'}),

        # Heatmap for recovery time vs. two features
        dcc.Graph(id='recovery-heatmap', style={'margin': '20px auto', 'width': '80%'}),

        # Download button
        html.Div([
            dbc.Button(
                "Download Results",
                id="download-button",
                className="mb-3",
                color="success",
                style={'width': '220px', 'margin': '20px auto', 'display': 'block'}
            ),
            dcc.Download(id="download-results")
        ]),
    ]
)

# Callback to toggle the metrics collapse
@app.callback(
    Output("metrics-collapse", "is_open"),
    [Input("metrics-collapse-button", "n_clicks")],
    [State("metrics-collapse", "is_open")],
)
def toggle_metrics(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback to toggle the legend collapse
@app.callback(
    Output("legend-collapse", "is_open"),
    [Input("legend-collapse-button", "n_clicks")],
    [State("legend-collapse", "is_open")],
)
def toggle_legend(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback to toggle the product description collapse
@app.callback(
    Output("product-collapse", "is_open"),
    [Input("product-collapse-button", "n_clicks")],
    [State("product-collapse", "is_open")],
)
def toggle_product_description(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# Callback for prediction, scatter plot, and heatmap
@app.callback(
    [Output('output-prediction', 'children'),
     Output('prediction-gauge', 'figure'),
     Output('recovery-progress', 'value'),
     Output('recovery-scatter-plot', 'figure'),
     Output('recovery-heatmap', 'figure')],  # Add heatmap output
    [Input('predict-button', 'n_clicks'),
     Input('scatter-feature', 'value')],  # Add scatter feature input
    [Input('age', 'value'), Input('duration', 'value'), Input('signs', 'value'),
     Input('temperature', 'value'), Input('respiratory_rate', 'value'), Input('time', 'value'),
     Input('clinical_score', 'value'), Input('km', 'value'), Input('hazard_function', 'value'),
     Input('drug_administered', 'value')]
)
def predict_recovery_time(n_clicks, scatter_feature, age, duration, signs, temperature, respiratory_rate, time,
                          clinical_score, km, hazard_function, drug_administered):
    if n_clicks > 0:
        drug_ceftriaxon = 1 if drug_administered == 'ceftriaxon' else 0
        drug_pencillin = 1 if drug_administered == 'cristaline_pencillin' else 0
        drug_ampicillin = 1 if drug_administered == 'ampicillin_gentamicin' else 0

        input_data = np.array([[
            age, duration, signs, temperature, respiratory_rate, time,
            clinical_score, km, hazard_function,
            drug_pencillin, drug_ceftriaxon, drug_ampicillin
        ]])

        recovery_time = model.predict(input_data)[0]

        # Generate scatter plot data
        feature_values = np.linspace(0, 1200, 100) if scatter_feature == 'age' else np.linspace(1, 30, 100) if scatter_feature == 'duration' else np.linspace(35, 42, 100) if scatter_feature == 'temperature' else np.linspace(10, 100, 100) if scatter_feature == 'respiratory_rate' else np.linspace(1, 100, 100) if scatter_feature == 'time' else np.linspace(0, 10, 100)
        recovery_times = [model.predict([[age if scatter_feature != 'age' else val,
                                          duration if scatter_feature != 'duration' else val,
                                          signs,
                                          temperature if scatter_feature != 'temperature' else val,
                                          respiratory_rate if scatter_feature != 'respiratory_rate' else val,
                                          time if scatter_feature != 'time' else val,
                                          clinical_score if scatter_feature != 'clinical_score' else val,
                                          km,
                                          hazard_function,
                                          drug_pencillin, drug_ceftriaxon, drug_ampicillin]])[0] for val in feature_values]
        scatter_data = pd.DataFrame({scatter_feature: feature_values, "Predicted Recovery Time (Days)": recovery_times})

        # Create scatter plot
        scatter_fig = px.scatter(
            scatter_data,
            x=scatter_feature,
            y="Predicted Recovery Time (Days)",
            title=f"Predicted Recovery Time vs. {scatter_feature}",
            trendline="lowess"
        )

        # Generate heatmap data
        heatmap_data = []
        for x in np.linspace(0, 1200, 20):  # Age
            row = []
            for y in np.linspace(1, 30, 20):  # Hospitalization Duration
                row.append(model.predict([[x, y, signs, temperature, respiratory_rate, time,
                                          clinical_score, km, hazard_function,
                                          drug_pencillin, drug_ceftriaxon, drug_ampicillin]])[0])
            heatmap_data.append(row)

        # Create heatmap
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=np.linspace(1, 30, 20),
            y=np.linspace(0, 1200, 20),
            colorscale='Viridis'
        ))
        heatmap_fig.update_layout(
            title="Recovery Time Heatmap (Age vs. Hospitalization Duration)",
            xaxis_title="Hospitalization Duration (Days)",
            yaxis_title="Age (Months)"
        )

        # Return the text, gauge figure, progress bar value, scatter plot, and heatmap
        return f'🩺 Estimated Recovery Time: {recovery_time:.2f} days', create_gauge(recovery_time), recovery_time, scatter_fig, heatmap_fig
    return "", go.Figure(), 0, go.Figure(), go.Figure()  # Return default values if no prediction is made

# Callback for downloading results
@app.callback(
    Output("download-results", "data"),
    [Input("download-button", "n_clicks")],
    [State('output-prediction', 'children'),
     State('prediction-gauge', 'figure'),
     State('recovery-scatter-plot', 'figure'),
     State('recovery-heatmap', 'figure')],
    prevent_initial_call=True
)
def download_results(n_clicks, prediction_text, gauge_fig, scatter_fig, heatmap_fig):
    if n_clicks:
        # Create a summary of the results
        summary = f"""
        Pneumonia Recovery Time Prediction Tool - Results

        {prediction_text}

        Visualizations:
        - Gauge Chart: Predicted recovery time.
        - Scatter Plot: Relationship between selected feature and recovery time.
        - Heatmap: Relationship between Age and Hospitalization Duration.

        Note: Visualizations are not included in this text file. Please refer to the app for interactive charts.
        """

        # Save the summary to a text file
        return dict(content=summary, filename="recovery_results.txt")

if __name__ == '__main__':
    app.run_server(debug=False, host="0.0.0.0", port=8080)
