library(readxl)
library(dynlm)
library(lmtest)
library(forecast)
library(dplyr)
library(ggplot2)

# Read in the data 
rm(list = ls())
setwd('C:/Users/PC/Documents/Tue course/Data Challange 2/JBG050-Data-Challenge-2-Group15')
df = read.csv('data/df_all.csv')
df = data.frame(df)[, -1]
date = df[, 1]

# Get total 
total <- list()
for(row in 1:nrow(df)) {
  total <- append(total, sum(df[row, -1]))  
}
df$total <- total

# Format time-series 
df_total = ts(df['Barnet.Vale'], start = c(2010, 12), frequency=12)

# Seasonal plot 
ggseasonplot(df_total, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("Crime") +
  ggtitle("Seasonal plot")

# Monthly smooth
dframe <- cbind(Monthly = df_total,
                DailyAverage = df_total/monthdays(df_total))
autoplot(dframe, facet=TRUE) +
  xlab("Years") + ylab("Pounds") +
  ggtitle("Crime per month/days in month")

# Case 
barnet.Vale <- window(df_total, end=2021)
h <- 10
fit.lin <- tslm(barnet.Vale ~ trend)
fcasts.lin <- forecast(fit.lin, h = h)
fit.exp <- tslm(barnet.Vale ~ trend, lambda = 0)
fcasts.exp <- forecast(fit.exp, h = h)

## Time break
t <- time(barnet.Vale)
t.break1 <- 2018
t.break2 <- 2020
tb1 <- ts(pmax(0, t - t.break1), start = 2010)
tb2 <- ts(pmax(0, t - t.break2), start = 2010)
fit.pw <- tslm(log(barnet.Vale) ~ t + tb1 + tb2)


## test dataset
test <- window(df_total, start=2021)
t.new <- time(test)
tb1.new <- ts(pmax(0, t.new - t.break1), start = 2021, frequency=12)
tb2.new <- ts(pmax(0, t.new - t.break2), start = 2021, frequency=12)
newdata <- cbind(t=t.new, tb1=tb1.new, tb2=tb2.new) %>%
  as.data.frame()
fcasts.pw <- forecast(fit.pw, newdata = newdata, h=12)

# More lags added + difference + back-transform 
total <- diff(df_total)
new_df <- window(total, end=c(2021, 12))
lag1 <- stats::lag(new_df, k=-1)
lag2 <- stats::lag(new_df, k=-2)
lag3 <- stats::lag(new_df, k=-3)
lag4 <- stats::lag(new_df, k=-4)
lag5 <- stats::lag(new_df, k=-5)
diff_df <- cbind(diff=new_df, lag1=lag1, lag2=lag2, lag3=lag3, lag4=lag4, lag5=lag5)
covid <- (time(diff_df) < 2018.9) | (time(diff_df) > 2021.3) # In covid or not 
in_covid <- ifelse(covid == TRUE, 1, 0)
diff_df <- cbind(diff_df, in_covid, t=time(diff_df))
fit.diff <- tslm(diff_df.diff ~ t + diff_df.lag1 + diff_df.lag2 + diff_df.lag3 + diff_df.lag4 + diff_df.lag5 + in_covid, data=diff_df)

# test data
test <- window(total, start=c(2022, 1))
lag1 <- stats::lag(test, k=-1)
lag2 <- stats::lag(test, k=-2)
lag3 <- stats::lag(test, k=-3)
lag4 <- stats::lag(test, k=-4)
lag5 <- stats::lag(test, k=-5)
diff_df <- cbind(diff=test, lag1=lag1, lag2=lag2, lag3=lag3, lag4=lag4, lag5=lag5)
covid <- (time(diff_df) < 2018.9) | (time(diff_df) > 2021.3) # In covid or not 
in_covid <- ifelse(covid == TRUE, 1, 0)
diff_df <- cbind(diff_df, in_covid, t=time(diff_df))
diff_df[, 8] = as.numeric(diff_df[, 8])
# Forcasting 
fcasts.pw <- forecast(fit.diff, newdata = diff_df, h=12)
