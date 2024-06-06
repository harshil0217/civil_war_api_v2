from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson') as response:
    countries = json.load(response)
import plotly.express as px
import pandas as pd


probs = pd.read_csv("./data/civil_war_prob.csv")

fig = px.choropleth(probs, geojson=countries, locations = 'country_text_id', color = 'civil_war_prob', 
                    locationmode='ISO-3', featureidkey= 'properties.ISO_A3', 
                    color_continuous_scale="Reds")

fig.update_geos(showcountries=True, showcoastlines=True, showland=False, fitbounds="locations")

fig.update_layout(
    geo=dict(
        showocean=True, oceancolor="LightBlue",
        showlakes=True, lakecolor="LightBlue"
    )
)

fig.show()