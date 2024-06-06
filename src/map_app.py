from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson') as response:
    countries = json.load(response)
import plotly.express as px
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


probs = pd.read_csv("./data/civil_war_prob.csv")

fig = px.choropleth(probs, geojson=countries, locations = 'country_text_id', color = 'civil_war_prob', 
                    locationmode='ISO-3', featureidkey= 'properties.ISO_A3', 
                    color_continuous_scale="Reds")

fig.update_geos(showframe=False)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("Civil War Probability by Country 2024"),
    dcc.Graph(figure=fig)
])

app.run_server(debug=True)