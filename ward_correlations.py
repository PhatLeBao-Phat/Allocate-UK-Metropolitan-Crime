import pandas as pd
import numpy as np

def demo_clean(df):
    df = df[df['Names'].str.contains('Barnet')].reset_index(drop=True)

    # Find the index of the row to be copied
    index_to_copy = df[df['Lower Super Output Area'] == 'E01000149'].index[0]

    # Copy the existing row and delete it
    new_row = df.loc[index_to_copy].copy()
    df = df.drop([index_to_copy])

    length = len(df)
    # Modify the column value for the new instances
    new_row['Lower Super Output Area'] = 'E01033572'
    df.loc[(length + 1)] = new_row

    new_row['Lower Super Output Area'] = 'E01033573'
    df.loc[(length + 2)] = new_row

    crime_columns = [col for col in df.columns if col.startswith('Crime')]
    df = df.drop(columns=crime_columns)
    return df

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

df_demo = pd.read_csv('lsoa-data-old-boundaries-DataSheet.csv', encoding= 'unicode_escape')
df_demo_clean = demo_clean(df_demo)
df_barnet = pd.read_csv('barnet.csv')
df_lsoa_burg = df_barnet.groupby('LSOA code').size().to_frame().rename(columns={0:'Burglaries'}).reset_index()
df_combined = pd.merge(df_lsoa_burg, df_demo_clean, left_on='LSOA code', right_on='Lower Super Output Area', how='inner')


lsoa_to_ward = pd.read_csv('LSOA_(2021)_to_Ward_to_Lower_Tier_Local_Authority_(May_2022)_Lookup_for_England_and_Wales (1).csv')
lsoa_to_ward = lsoa_to_ward[['LSOA21CD', 'WD22CD', 'WD22NM']]
unknown = []
for code in df_combined['LSOA code']:
    if code not in lsoa_to_ward['LSOA21CD'].to_list():
        unknown.append(code)
lsoa_to_ward = to_wards(lsoa_to_ward, unknown)
df_combined = add_wards(df_combined, lsoa_to_ward)
ward_code = df_combined.pop('Ward Code')
df_combined = df_combined.select_dtypes(include=[np.number])
df_combined['Ward Code'] = ward_code

ward_features = df_combined.groupby('Ward Code').mean()
correlations = ward_features.corr()['Burglaries'].abs().sort_values(ascending=False)
correlations = correlations.drop('Burglaries')
# Remove 'Burglaries' itself from the list
print(correlations[:30])