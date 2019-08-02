args = commandArgs(trailingOnly=TRUE)

library(survival)
library(broom)

file_name <- args[1]

df <- read.csv(file_name, stringsAsFactors=FALSE)

TTP_times <- df$TTP_times
events <- as.logical(df$events)
treatment_arms <- df$treatment_arms

fit <- tidy( coxph( Surv(TTP_times, events) ~ treatment_arms, data=data.frame(treatment_arms) ) )

print( fit$estimate[1] )

