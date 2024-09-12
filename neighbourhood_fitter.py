import pandas as pd
import plotly.express as px
# from cleaning import AB_data
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import tqdm
import json

# For tracking progress because it takes a while to complete
tqdm.pandas()

# Neighbourhoods polygons (for plotting). This is the additional dataset
with open("data/nyc-neighborhoods.geo.json", "r") as geojsonfile:
    geojson = json.load(geojsonfile)

# Our algorithm that assigns a neighbourhood to the accomodations, otherwise "not found"
def assign_neighbourhood(x, y):
    point = Point(x,y)
    name_value = "not found"
    for i in geojson['features']:
        name = i['properties']['name']
        raw_polygon = i['geometry']['coordinates'][0]
        if type(raw_polygon[0][0]) == float:
            clean_polygon = [tuple(list(reversed(j))) for j in raw_polygon]
        else:
            clean_polygon = [tuple(list(reversed(j))) for j in raw_polygon[0]]
        polygon = Polygon(clean_polygon)
        if polygon.contains(point):
            name_value = name
    return name_value

# Adds new neighbourhood column
AB_data['fitted_neighbourhood'] = AB_data.progress_apply(lambda row:assign_neighbourhood(row.lat, row.long), axis=1)

# Save pickle file
AB_data.to_pickle('blablabla.pickle')



