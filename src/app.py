import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the CSV data from the provided link
url = "https://media.githubusercontent.com/media/sunshineluyao/icp-nns-db/main/data/proposals_no_empty.csv"
df = pd.read_csv(url)

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Create a bar plot using Plotly Express with a log-scale y-axis
fig = px.bar(df['action'].value_counts(), x=df['action'].value_counts().index, y=df['action'].value_counts().values, labels={'x': 'Action', 'y': 'Count'}, title='Distribution of Actions')
fig.update_yaxes(type="log")  # Set y-axis to logarithmic scale

# Define the layout of the dashboard
app.layout = html.Div([
    # Header
    html.H1("Internet Computer Protocol NNS Governance System Dashboard: Action"),
    
    # Paragraph with italicized notes
    html.P([
        "Choose the proposal status to view the distributions of proposal actions. ",
        html.I("Notes: Action is the type of action the proposal seeks, such as propose, reject, or execute; Status is the current standing of the proposal, be it pending, accepted, negated, or unsuccessful.")
    ]),

    # Dropdown for status selection
    dcc.Dropdown(
        id='status-dropdown',
        options=[{'label': status, 'value': status} for status in df['status'].unique()],
        value=df['status'].unique()[0],  # Set default value to the first unique status
        multi=False
    ),

    # Graph for displaying the data
    dcc.Graph(id='bar-plot')
])

# Create a callback to update the bar plot based on the selected status
@app.callback(
    Output('bar-plot', 'figure'),
    Input('status-dropdown', 'value')
)
def update_bar_plot(selected_status):
    filtered_df = df[df['status'] == selected_status]
    fig = px.bar(filtered_df['action'].value_counts(), x=filtered_df['action'].value_counts().index,y=filtered_df['action'].value_counts().values, labels={'x': 'Action', 'y': 'Count'},title=f'Distribution of Actions for {selected_status} Proposals')
    fig.update_yaxes(type="log")  # Set y-axis to logarithmic scale
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
