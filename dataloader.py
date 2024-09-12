import pandas as pd

# To create a dataframe with the 'street' files
crime_df = pd.concat([
    pd.read_csv(f'crime_data/{year}-{str(month).zfill(2)}/{year}-{str(month).zfill(2)}-metropolitan-street.csv')
    for year, months in ((2010, range(12, 13)), *( (year, range(1, 13)) for year in range(2011, 2023)), (2023, range(1, 3)))
    for month in months
])

# To create a dataframe with the 'outcomes' files
outcomes_df = pd.concat([
    pd.read_csv(f'outcome_and_stopsearch_data/{year}-{str(month).zfill(2)}/{year}-{str(month).zfill(2)}-metropolitan-outcomes.csv')
    for year, months in ((2020, range(4, 13)), *( (year, range(1, 13)) for year in range(2021, 2023)), (2023, range(1, 4)))
    for month in months
])

# To create a dataframe with the 'stop-and-search' files
stopsearch_df = pd.concat([
    pd.read_csv(f'outcome_and_stopsearch_data/{year}-{str(month).zfill(2)}/{year}-{str(month).zfill(2)}-metropolitan-stop-and-search.csv')
    for year, months in ((2020, range(4, 13)), *( (year, range(1, 13)) for year in range(2021, 2023)), (2023, range(1, 4)))
    for month in months if not (month == 11 and year == 2022) # File of November 2022 does not exist somehow
])
