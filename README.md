# Data Challenge 2: Allocating police demand for residential burglaries

This repository holds the source code for the development of the tool for JBG050-DC2.
We've created a model that predicts the number of burglaries in a given area of Barnet (a borough of London), based on
the number of burglaries in the past.

* The `main` branch holds the files that were created during the development of the final model and exploration of the
  datasets.
* The `final_dashboard` branch holds the files containing the actual final tool (dashboard).
* The `extra-analysis` branch can be disregarded. It contains some extra analysis that did not make it to the
  production-level code.
* The `phat-model` branch can also be disregarded. It also contains files that were created during the development of
  the model.

## Code structure

The code is structured into multiple files, based on their functionality.
The files are:

| File                                | Description                                                                                                                                 | Remarks                                                                                                                            |
|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `barnet_fitter.py`                  | Generates the `burg_ward_map.html` file that shows an elementary implementation (proof of concept) of one of the plots in our dashboard.    | It fits a new model for each ward (subarea) of Barnet and visualizes the predictions in a geographical manner.                     |
| `barnet.csv`                        | This is the cleaned dataset that originates from the official data provided by us.                                                          | None                                                                                                                               |
| `dataloader.py`                     | This loads all of the official data provided into memory.                                                                                   | We eventually used Barnet's data only.                                                                                             |
| `download_data.py`                  | This file contains code that can pull the most recent official data.                                                                        | This functionality did not make it into production and was abandoned in the development phase (because we didn't go to London). :( |
| `EDA.ipynb`                         | A file that contains some exploratory data analysis about the dataset.                                                                      | Feel free to have a *scroll* (not a run!) through this file, as this file is not necessary for the final product.                  |
| `feature_corr_ward.py`              | Presents the 30 attributes that have the highest correlation with amount of burglaries.                                                     | None                                                                                                                               |
| `feature_selection_lsoa_dataset.py` | Presents the 30 attributes that have the highest correlation with amount of burglaries.                                                     | Same as the file above, but grouped by LSOA instead of ward.                                                                       |
| `neighbourhood_fitter.py`           | This file contains template code from a previous project that assigns single crimes to a ward based on their latitude and longitude values. | Did not make it to production.                                                                                                     |
| `time_series.ipynb`                 | Final model was developed here along with a lot of EDA with regards to model development.                                                   | Can run, but still recommended to scroll through as the plots are already generated.                                               |
| `weather.py`                        | Explorations for checking whether we could use weather data to assist with predicting the amount of burglaries.                             | Did not make it to production (weak predictive capabilities).                                                                      |

## Dashboard

Our final tool is a dashboard that can be used to visualize the police demand in any area of Barnet, based on the number
of burglaries in the past. The dashboard is built using the Dash framework. The dashboard can be found in
the `final_dashboard` branch. The dashboard can be run by running the `app.py` file in the `dashboard` directory.
After running app.py you can open the dashboard by going to the link provided in the terminal.

Some interesting files and directories that are only in the `final_dashboard` branch are:

| File or directory  | Description                                                                        | Remarks                                              |
|--------------------|------------------------------------------------------------------------------------|------------------------------------------------------|
| `lineplots/`       | Contains the lineplots that are used to update the lineplot view in the dashboard. | None                                                 |
| `weatherdata/`     | Contains some of the plots related to the exploration of weather data.             | Not relevant as it did not make it to production.    |
| `dashboard/app.py` | File that starts up the entire dashboard.                                          | See note 3 at the bottom of this documentation file. |

## Environment setup instructions

We recommend to set up a virtual Python environment and use a common IDE like Pycharm to install the package and its
dependencies. To install the package, we recommend to execute the following command in the command line:

```
pip install -r requirements.txt
```

If you are using PyCharm, it should offer you the option to create a virtual environment from the requirements file on
startup upon your first time opening the file.

**Note 1:** We used Python 3.9 for this project. Some libraries that were used do not support newer versions of Python
at this point in time.

**Note 2:** We left several files and directories in the repository that are not used in the final product. This is
because we wanted to retain some knowledge that was gained through stuff that didn't make it into production. We
recommend to ignore these files and directories.

**Note 3:** If the dashboard initially doesn't show some of the visualizations, choose a value in both dropdown menus
from the panel to the left.

**Note 4:** When you want to run the code yourself, we recommend cloning the `final_dashboard` branch instead of the `main` (default) branch, as this branch contains the final product.
