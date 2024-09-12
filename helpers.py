"""Helpers function to load data"""
import os 
import numpy as np 
import pandas as pd 

# Path to ward.csv 
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data\LSOAs/LSOA_(2021)_to_Ward_to_Lower_Tier_Local_Authority_(May_2022)_Lookup_for_England_and_Wales.csv")
wards = pd.read_csv(path)
wards = wards[['LSOA21CD', 'WD22NM']]
wards['LSOA code'] = wards['LSOA21CD']
wards.drop(columns=['LSOA21CD'])


def mapping_wards(df: pd.DataFrame, wards: pd.DataFrame = wards, reset_index: bool = True) -> pd.DataFrame:
    """
    Mapping to wards. reset_index(drop=True)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with a column name is LSOA code.
    wards: pd.DataFrame
        Dataframe maps LSOAs (LSOA21CD) to wards (WD22NM).

    Returns
    -------
    pd.DataFrame
        df with 1 extra column name WD22NM.
    """
    df = df.reset_index(drop=reset_index)
    df_mapped = pd.merge(
        left=df,
        right=wards,
        on='LSOA code',
        how='left',
    )
    df_mapped.drop(columns=['LSOA21CD'], inplace=True)

    return df_mapped


def count_per_wards(
    df, 
    values_column: str = 'LSOA_code', 
    index_column: str = 'Month',
    columns: str = 'ward',
    reset_index: bool=True) -> pd.DataFrame:
    """
    Pivot transformation. turn row index as time and column as wards. 
    The value in square is count of crimes. 
    
    Parameters
    ----------
    df : pd.DataFrame
    values_column : str, optional, default='LSOA_code'
        A column used to count. 
    index_column : str, optional, default='Month'
        A column used to set the index of row.
    columns : str, optional, default='ward'
        Column denotes the name of wards. 
    reset_index: bool, optional=[True, False]
    Returns
    -------
    pd.DataFrame
    """
    df_all = pd.pivot_table(df, values='Year', index='Time', columns='WD22NM', aggfunc='count')
    df_all.columns.name = None
    df_all = df_all.reset_index()
    
    return df_all