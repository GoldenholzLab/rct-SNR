args = commandArgs(trailingOnly=TRUE)

library(powerSurvEpi)

nE    =    strtoi(args[1])
nC    =    strtoi(args[2])
pE    = as.double(args[3])
pC    = as.double(args[4])
RR    = as.double(args[5])
alpha = as.double(args[6])

power <- powerCT.default(nE, nC, pE, pC, RR, alpha)

print(100*power)