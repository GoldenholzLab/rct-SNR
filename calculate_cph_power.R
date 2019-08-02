args = commandArgs(trailingOnly=TRUE)

library(survival)
library(broom)
library(powerSurvEpi)

file_name <- args[1]
alpha     <- as.double(args[2])
numC      <- strtoi(args[3])
numE      <- strtoi(args[4])

df <- read.csv(file_name, stringsAsFactors=FALSE)

TTP_times <- df$TTP_times
events <- as.logical(df$events)
treatment_arms <- df$treatment_arms
f_treatment_arms <- factor(df$treatment_arms_str)

fit <- tidy( coxph( Surv(TTP_times, events) ~ treatment_arms, data=data.frame(treatment_arms) ) )
plhr <- fit$estimate[1]
values <- powerCT( Surv(TTP_times, events) ~ f_treatment_arms, data.frame(TTP_times, events, f_treatment_arms), numE, numC, plhr, alpha)

print( plhr )
print( 100*values$pC )
print( 100*values$pE )
print( 100*values$power )