library(lmtest)
library(forecast)
library(dplyr)
library(ggplot2)
library(rstudioapi) 
library(glue)
# Remove env var 
rm(list = ls())

# Read the dataset 
getSourceEditorContext()$path %>% dirname() %>% dirname() %>% setwd()
df <- read.csv('data/df_all.csv')
ts_df <- ts(df, start = c(2010, 12), frequency=12)
wards_name <- dimnames(df)[[2]][-c(1, 2)]

# Function to predict 
predict <- function(ts_df, ward, stepwise=TRUE, approximation=TRUE, h=12){
  df <- ts_df[, ward]
  # Fit 
  glue('Fitting {ward}...') %>% print()
  fit <- auto.arima(df, seasonal = TRUE, stepwise=stepwise, approximation=approximation)
  # Check with noise 
  print('Checking white noise...')
  checkresiduals(fit, plot=FALSE)$p.value %>% print()
  # Forecast
  predicted <- fit %>% forecast(h=h)
  # browser()
  return(predicted$mean)
}

# Main:
predicted <- matrix(predict(ts_df, wards_name[1], stepwise = FALSE, approximation = FALSE))
for(ward in wards_name[-1]){
  predicted <- cbind(predicted, predict(ts_df, ward, stepwise = FALSE, approximation = FALSE))
}
colnames(predicted) <- wards_name

# write out the file 
write.csv(predicted, file='data/total_crime_predicted.csv')
