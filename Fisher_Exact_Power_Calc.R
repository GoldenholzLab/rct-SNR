args = commandArgs(trailingOnly=TRUE)

library(statmod)

num_placebo <- as.double(args[1])
num_drug <- as.double(args[2])
expected_placebo_response <- as.double(args[3])/100
expected_drug_response <- as.double(args[4])/100

power_estimate <- 100*power.fisher.test(expected_placebo_response, expected_drug_response, num_placebo, num_drug)

print(power_estimate)