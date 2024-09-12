import pandas as pd
import json
import geopandas as gpd
import folium
from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiPolygon
from statsmodels.tsa.arima.model import ARIMA

def to_wards(df, list):
    '''
    Function to add the missing LSOA-ward combinations
    to the lsoa_to_ward dataset.
    '''
    # Add the new lsoacode-wardcode-wardname combinations as new row in lsoa_to_ward
    index = len(df)
    for lsoa in list:
        if lsoa == 'E01000125':
            code = df[df['WD22NM'] == 'Burnt Oak']['WD22CD'].values[0]
            df.loc[index] = [lsoa, code, 'Burnt Oak']
            index += 1
            continue
        if lsoa == 'E01000262': # This lsoa has been assigned to Totteridge, although could have also been Mill Hill
            code = df[df['WD22NM'] == 'Totteridge & Woodside']['WD22CD'].values[0]
            df.loc[index] = [lsoa, code, 'Totteridge & Woodside']
            index += 1
            continue
        else:
            code = df[df['WD22NM'] == 'Colindale South']['WD22CD'].values[0] # E01000155 could have also been added to North instead of South
            df.loc[index] = [lsoa, code, 'Colindale South']
            index += 1
    return df

def add_wards(df_barnet, df_lsoa_ward):
    '''
    Function to add the ward's name and code to the original
    barnet dataset
    '''
    # Map every LSOA code to a ward code and name
    lsoa_mapping = {}
    for index, row in df_lsoa_ward.iterrows():
        lsoa_code = row['LSOA21CD']
        ward_code = row['WD22CD']
        ward_name = row['WD22NM']
        lsoa_mapping[lsoa_code] = (ward_code, ward_name)

    # Process each LSOA code
    codes = []
    names = []
    for lsoa_code in df_barnet['LSOA code']:
        if lsoa_code in lsoa_mapping:
            ward_code, ward_name = lsoa_mapping[lsoa_code]
            codes.append(ward_code)
            names.append(ward_name)

    # Add the ward codes and names to the barnet dataset
    df_barnet['Ward Code'] = codes
    df_barnet['Ward Name'] = names
    return df_barnet

def split_df(barnet):
    '''
    Sybren's function to create 24 dataframes for every ward
    in Barnet with the original barnet data
    '''
    # Split the total DataFrame into DataFrame's for each ward
    df_barnet_vale = barnet.loc[barnet['Ward Name'] == 'Barnet Vale']
    df_brunswick = barnet.loc[barnet['Ward Name'] == 'Brunswick Park']
    df_burnt_oak = barnet.loc[barnet['Ward Name'] == 'Burnt Oak']
    df_childs_hill = barnet.loc[barnet['Ward Name'] == 'Childs Hill']
    df_colindale_north = barnet.loc[barnet['Ward Name'] == 'Colindale North']
    df_colindale_south = barnet.loc[barnet['Ward Name'] == 'Colindale South']
    df_cricklewood = barnet.loc[barnet['Ward Name'] == 'Cricklewood']
    df_east_barnet = barnet.loc[barnet['Ward Name'] == 'East Barnet']
    df_east_finchley = barnet.loc[barnet['Ward Name'] == 'East Finchley']
    df_edgware = barnet.loc[barnet['Ward Name'] == 'Edgware']
    df_edgwarebury = barnet.loc[barnet['Ward Name'] == 'Edgwarebury']
    df_finchy = barnet.loc[barnet['Ward Name'] == 'Finchley Church End']
    df_friern = barnet.loc[barnet['Ward Name'] == 'Friern Barnet']
    df_garden_suburb = barnet.loc[barnet['Ward Name'] == 'Garden Suburb']
    df_golders_green = barnet.loc[barnet['Ward Name'] == 'Golders Green']
    df_hendon = barnet.loc[barnet['Ward Name'] == 'Hendon']
    df_high_barnet = barnet.loc[barnet['Ward Name'] == 'High Barnet']
    df_mill_hill = barnet.loc[barnet['Ward Name'] == 'Mill Hill']
    df_totteridge = barnet.loc[barnet['Ward Name'] == 'Totteridge & Woodside']
    df_underhill = barnet.loc[barnet['Ward Name'] == 'Underhill']
    df_west_finchley = barnet.loc[barnet['Ward Name'] == 'West Finchley']
    df_west_hendon = barnet.loc[barnet['Ward Name'] == 'West Hendon']
    df_whetstone = barnet.loc[barnet['Ward Name'] == 'Whetstone']
    df_woodhouse = barnet.loc[barnet['Ward Name'] == 'Woodhouse']

    df_wards = [df_barnet_vale, df_brunswick, df_burnt_oak, df_childs_hill, df_colindale_north, df_colindale_south,
                df_cricklewood, df_east_barnet, df_east_finchley, df_edgware, df_edgwarebury, df_finchy, df_friern,
                df_garden_suburb, df_golders_green, df_hendon, df_high_barnet, df_mill_hill, df_totteridge,
                df_underhill,
                df_west_finchley, df_west_hendon, df_whetstone, df_woodhouse]
    return df_wards

def predictions(list_df):
    '''
    Sybren's function (slightly changed) to obtain the predictions based
    on the ARIMA model
    '''
    df_predictions = pd.DataFrame()
    # For every ward dataframe create a separate model for predictions
    for df in list_df:
        # Obtain the burglaries per month for the given ward df
        temp = pd.DataFrame()
        temp['freq'] = df['Month'].value_counts()
        temp = temp.reset_index().rename(columns={'index':'Month'}).sort_values(by=['Month']).reset_index().drop(columns=['index'])
        temp['Month'] = pd.to_datetime(temp['Month'])

        # This is needed, because otherwise months could be missing
        # Generate a range of all months between the minimum and maximum dates in the dataframe
        min_date = temp['Month'].min()
        max_date = temp['Month'].max()
        all_months = pd.date_range(start=min_date, end=max_date, freq='MS')

        # Create a new dataframe with all months and merge it with the original dataframe
        new_df = pd.DataFrame({'Month': all_months})
        temp = pd.merge(temp, new_df, on='Month', how='right')

        # Fill missing frequencies with 0
        temp['freq'].fillna(0, inplace=True)

        # Sort the dataframe by months
        temp.sort_values('Month', inplace=True)

        # Reset the index if needed
        temp.reset_index(drop=True, inplace=True)
        temp.set_index('Month', inplace = True)
        # Train model on the DateTime and obtain prediction for next month
        model = ARIMA(temp, order=(11,1,1), seasonal_order=(0,1,1,12))
        result = model.fit()
        predict = result.forecast(steps=1)
        code = df['Ward Code'].iloc[0]
        df_predictions[code] = predict.values
    return df_predictions

def convert(df_pred, lsoa_to_ward):
    '''
    Function to go from dataframe with predictions per
    Ward code, to a dataframe with predictions, predictions
    as percentage of total burglaries, ward codes and ward names
    '''

    # Create new dataframe to store predictions in
    df = pd.DataFrame()
    total_pred = sum(df_pred.values[0])
    df['Burglaries'] = df_pred.values[0].round(decimals=2)
    df['Percentage'] = ((df_pred.values[0] / total_pred)*100).round(decimals=2)
    df['WD22CD'] = df_pred.columns.to_list()
    names = []
    # Find ward names corresponding to the ward codes
    for code in df['WD22CD']:
        name = lsoa_to_ward[lsoa_to_ward['WD22CD'] == code]['WD22NM'].values[0]
        names.append(name)
    df['WD22NM'] = names
    return df

# Load in data
barnet = pd.read_csv('barnet.csv').drop(columns=['Unnamed: 0', 'index', 'Unnamed: 0.1'])
barnet_burg = barnet.groupby('LSOA code').size().reset_index().rename(columns={0:'Burglaries'})
lsoa_to_ward = pd.read_csv('LSOA_(2021)_to_Ward_to_Lower_Tier_Local_Authority_(May_2022)_Lookup_for_England_and_Wales (1).csv')
lsoa_to_ward = lsoa_to_ward[['LSOA21CD', 'WD22CD', 'WD22NM']]

# Find LSOAs that are in barnet dataset but not in the conversion dataframe
unknown = []
for i in barnet_burg['LSOA code']:
    if i not in lsoa_to_ward['LSOA21CD'].to_list():
        unknown.append(i)

# Code to obtain the predictions for next month
# Add the unknown LSOAs to the conversion dataframe and use that to add wards to existing barnet dataframe
lsoa_to_ward = to_wards(lsoa_to_ward, unknown)
barnet = add_wards(barnet, lsoa_to_ward)
# Do the predictions per ward and store in df_ward
list_wards_df = split_df(barnet)
df_pred = predictions(list_wards_df)
df_ward = convert(df_pred, lsoa_to_ward)

# For choropleth with the total burglaries per ward, not for predictions therefore commented out
# df_merged = pd.merge(barnet_burg, lsoa_to_ward, left_on='LSOA code', right_on='LSOA21CD')
# df_ward = df_merged[['WD22CD', 'WD22NM', 'Burglaries']]
# df_ward = df_ward.groupby(['WD22CD', 'WD22NM']).sum().reset_index()

# Choropleth creation
# Some initializations
map_center = [51.5074, -0.1278]
fig = folium.Map(width=900, height=600, zoom_start=10, location=map_center)
choropleth = {"features" : [], "type":"FeatureCollection"}

# Add the id's to the geojson data, so we can use key_on later for the choropleth
for i in range(28,52):
    code = 'E050136' + str(i)
    ward_boundary = json.load(open('barnet_geojson/AnyConv.com__' + code + '.geojson'))
    ward_boundary['features'][0]['properties'] = {'id' : code} # Add the id to the properties for the choropleth later
    choropleth['features'].append(ward_boundary['features'][0])

# Write the choropleth dictionary back to a geojson file, because geodata has to be changed from LineString to Multipolygon
json_object = json.dumps(choropleth)
with open('data/choropleth.geojson', 'w') as file:
    file.write(json_object)

# Load the json file back into a geopandas dataframe
df_geodata = gpd.read_file('data/choropleth.geojson')
geometry_list = []

# Code to change the geodata from type LineString to a MultiPolygon (so color can be filled for choropleth)
for multi_line in df_geodata['geometry']:
    border_lines = unary_union(multi_line)
    result = MultiPolygon(polygonize(border_lines))
    geometry_list.append(result)
df_geodata['geometry'] = geometry_list

# we rename the column from id to WD22CD so we can merge the two data frames.
geoJSON_df = df_geodata.rename(columns = {"id":"WD22CD"})
# Next we merge df_ward and the geoJSON data frame on the key id.
final_df = geoJSON_df.merge(df_ward, on = "WD22CD")

# Create the Choropleth based on percentage ward's burglaries to total burglaries Barnet
folium.Choropleth(
    geo_data=final_df,
    data=final_df,
    name='choropleth',
    columns=['WD22CD', 'Percentage'],
    key_on="feature.properties.WD22CD",
    fill_color='PuBu',
    legend_name='Percentage of total predicted burglaries',
    highlight=True,
    show=True,
    overlay=True
).add_to(fig)

# Make the highlighting a bit fancier
style_function = lambda x: {'fillColor': '#ffffff',
                            'color':'#000000',
                            'fillOpacity': 0.1,
                            'weight': 0.1}
highlight_function = lambda x: {'fillColor': '#000000',
                                'color':'#000000',
                                'fillOpacity': 0.50,
                                'weight': 0.1}

# Highlighting function to show additional data about the ward
NIL = folium.features.GeoJson(
    data = final_df,
    style_function=style_function,
    control=False,
    highlight_function=highlight_function,
    tooltip=folium.features.GeoJsonTooltip(
        fields=['WD22NM', 'WD22CD', 'Burglaries','Percentage'],
        aliases=['Ward Name', 'Ward Code','Predicted Burglaries','Predicted Burglaries (%)'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    )
)
fig.add_child(NIL)
fig.keep_in_front(NIL)

fig.save('burg_ward_map.html')