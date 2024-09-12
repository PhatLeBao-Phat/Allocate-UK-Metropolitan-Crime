import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from statistics import mean
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller
from gluonts.torch import DeepAREstimator
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator
from gluonts.mx.trainer import Trainer


df_weather_d = pd.read_csv("weatherdata/weather_daily.csv")
for i in df_weather_d.index:
    df_weather_d.loc[i, "month"] = df_weather_d.loc[i, "time"][:7]

# df_weather_h = pd.read_csv("weatherdata/weather_hourly.csv")
# print(df_weather_h.columns)
# for i in df_weather_h.index:
#     df_weather_h.loc[i, "month"] = df_weather_h.loc[i, "time"][:7]

df_months = pd.DataFrame()
for month in list(df_weather_d["month"]):
    df_current_month = df_weather_d[df_weather_d["month"] == month]
    df_months.loc[month, "rain_avg_curr"] = mean(list(df_current_month["rain_sum (mm)"]))  # mm
    df_months.loc[month, "temp_avg_curr"] = mean(list(df_current_month["temperature_2m_mean (°C)"]))  # °C
    df_months.loc[month, "snow_avg_curr"] = mean(list(df_current_month["snowfall_sum (cm)"]))
    if df_months.loc[month, "snow_avg_curr"] > 0:
        df_months.loc[month, "snow_avg_curr"] = 1 # cm or 0/1 for No snow/Snow

df_crime = pd.read_csv("barnet.csv")
for month in df_months.index[1:]:
    df_months.loc[month, "crime"] = len(df_crime[df_crime["Month"] == month])
    df_months.loc[month, "rain_avg"] = df_months.loc[df_months.index[df_months.index.get_loc(month) - 1], "rain_avg_curr"]
    df_months.loc[month, "temp_avg"] = df_months.loc[df_months.index[df_months.index.get_loc(month) - 1], "temp_avg_curr"]
    df_months.loc[month, "snow_avg"] = df_months.loc[df_months.index[df_months.index.get_loc(month) - 1], "snow_avg_curr"]

df_months.index = pd.to_datetime(df_months.index)
df_months = df_months[11:]
train_size = int(len(df_months) * 0.8)  # 80% for training, 20% for testing
train_data = df_months[:train_size]
test_data = df_months[train_size:-2]

train_ds = ListDataset(
    [{"start": train_data.index[0], "target": train_data["crime"].values}],
    freq="M",
)
test_ds = ListDataset(
    [{"start": test_data.index[0], "target": test_data["crime"].values}],
    freq="M",
)

# Define and train the model
estimator = DeepAREstimator(
    prediction_length=12,  # Adjust as needed
    context_length=12,  # Adjust as needed
    freq="M",
    num_layers=2,  # Number of layers in the network
    hidden_size=64,  # Number of cells in each layer
    dropout_rate=0.1,
    trainer_kwargs={'max_epochs': 60}
)

predictor = estimator.train(train_ds, validation_data=train_ds,
                            ckpt_path="lightning_logs/version_25/checkpoints/epoch=33-step=1700.ckpt")

# Make predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=3000,  # Number of samples for probabilistic forecasts
)



forecasts = list(forecast_it)
# Evaluate the model
#evaluator = Evaluator()
#agg_metrics, item_metrics = evaluator(iter(forecasts), num_series=len(test_ds))

# Plot results
# for series in train_ds:
#     series.plot()
# plt.grid(which="both")
# plt.legend(["Actual"])
# plt.show()
plt.plot(test_data.index, test_data["crime"], label="Actual", color="green")
for forecast in forecasts:
    forecast.plot()
plt.grid(which="both")
plt.legend(["Actual", "Forecast"])
plt.savefig("weatherdata/gluonts.png")
# model = Prophet()
# result = model.fit(train_data)
# train_predictions = result.predict(start=train_data.index[0], end=train_data.index[-1])
# test_predictions = result.predict(start=test_data.index[0], end=test_data.index[-1])
# #print(result.summary())
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(df_months.index, df_months["crime"], label="Actual", color="blue")
# ax.plot(train_data.index, train_predictions, label="Train Predicted", color="red")
# ax.plot(test_data.index, test_predictions, label="Test Predicted", color="green")
# ax.set_xlabel("Date")
# ax.set_ylabel("Crime")
# ax.legend(["Actual", "Predicted (train set)", "Forecast"])
# plt.savefig("weatherdata/fbprophet.png")


model_rain = LinearRegression()
model_rain.fit(df_months["rain_avg"].values.reshape(-1, 1), list(df_months["crime"]))
model_snow = LinearRegression()
model_snow.fit(df_months["snow_avg"].values.reshape(-1, 1), list(df_months["crime"]))
model_temp = LinearRegression()
model_temp.fit(df_months["temp_avg"].values.reshape(-1, 1), list(df_months["crime"]))
plt.plot(df_months.index, model_rain.predict(df_months["rain_avg"].values.reshape(-1, 1)), color='blue')
plt.plot(df_months.index, model_snow.predict(df_months["snow_avg"].values.reshape(-1, 1)), color='black')
plt.plot(df_months.index, model_temp.predict(df_months["temp_avg"].values.reshape(-1, 1)), color='red')
plt.plot(df_months.index, df_months["crime"], color='green')
plt.xlabel("Month")
plt.ylabel('Crime')
plt.title(f'Predicted crime')
plt.legend(["rain", "snow", "temp", "actual"])
plt.savefig('weatherdata/pred.png')
plt.clf()

model = LinearRegression()
model.fit(df_months[["rain_avg", "snow_avg", "temp_avg"]], list(df_months["crime"]))
plt.plot(df_months.index, model.predict(df_months[["rain_avg", "snow_avg", "temp_avg"]]), color='red')
plt.plot(df_months.index, df_months["crime"], color='blue')
plt.xlabel("Month")
plt.ylabel('Crime')
plt.title(f'Predicted crime')
plt.legend(["predicted", "actual"])
plt.savefig('weatherdata/pred_all.png')
plt.clf()

# Plotting rain_avg vs. crime
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.scatter(df_months["crime"], df_months['rain_avg'], c='blue')
axs.set_xlabel('Burglaries')
axs.set_ylabel('Rain Avg')
axs.set_title('Rain Avg vs. Crime')
plt.savefig('weatherdata/scatter_rain.png')
plt.clf()

# Plotting temp_avg vs. crime
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.scatter(df_months["crime"], df_months['temp_avg'], c='red')
axs.set_xlabel('Burglaries')
axs.set_ylabel('Temp Avg')
axs.set_title('Temp Avg vs. Crime')
plt.savefig('weatherdata/scatter_temp.png')
plt.clf()

# Plotting snow_avg vs. crime
fig, axs = plt.subplots(1, 1, figsize=(8, 8))
axs.scatter(df_months["crime"], df_months['snow_avg'], c='green')
axs.set_xlabel('Burglaries')
axs.set_ylabel('Snow Avg')
axs.set_title('Snow Avg vs. Crime')
plt.savefig('weatherdata/scatter_snow.png')
plt.clf()



