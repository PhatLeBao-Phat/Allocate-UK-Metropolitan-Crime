library(readxl)
library(dynlm)
library(lmtest)
library(forecast)
library(urca)

# Read in the data 
rm(list = ls())
setwd('C:/Users/PC/Documents/Tue course/Data Challange 2/JBG050-Data-Challenge-2-Group15')

df = read.csv('Data/streets_total_count.csv')
df = data.frame(df)[, -1]
date = df[, 1]

# Format time-series 
df_total = ts(df[, 2], start = c(2020, 4), frequency=12)

#Plot
plot(df_total, col="red", xlab="Crimes count (burglary)", ylab="")

# 1(d) Plot the ACF and PACF
par(mfrow=c(2,1))
acfpl <- acf(df_total, plot=TRUE)
pacfpl <- pacf(df_total, plot=TRUE)
# KPSS test
df_total %>% ur.kpss() %>% summary()
diff(df_total) %>% ur.kpss() %>% summary()
# 1(d) Plot the ACF and PACF of first diff 
par(mfrow=c(2,1))
acfpl <- acf(diff(df_total), plot=TRUE)
pacfpl <- pacf(diff(df_total), plot=TRUE)

# Transform the lags from years to months
acfpl$lag <- acfpl$lag * 12
pacfpl$lag <- pacfpl$lag * 12

# Plot the ACF
plot(acfpl, xlab="Lag (months)", main="", col="red", pch=1, ylab="")
title("Crime count (burglary)")

# Plot the PACF
plot(pacfpl, xlab="Lag (months)", main="", col="red", pch=2, ylab="")
title("Crime count (burglary)")

# Fit ARIMA model
df_total_copy = head(df_total, 33)
model = arima(df_total_copy, order = c(3,1,2), method = "CSS") # OLS
coeftest(model)

# Predict 
forecast::forecast(model, 5)
tail(df, 3)
autoplot(forecast::forecast(model, 5))

