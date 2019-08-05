
library(muhaz)
library(broom)

TTP_times <- c( 21, 15, 45, 85)
events    <- c(  1,  1,  1,  0)

muhaz_obj <- muhaz(TTP_times, events, min.time=0, max.time=84, bw.grid=c(10,20,30,30,50,60,70,80))

muhaz_summary <- summary( muhaz_obj )

print( muhaz_summary )
print(muhaz_obj)
print(muhaz_obj$haz.est)
print(muhaz_obj$est.grid)