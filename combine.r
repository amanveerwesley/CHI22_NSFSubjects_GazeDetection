library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

lf = list.files('CSV/Subjects/')

d = data.frame()
for (i in 1:length(lf)) {
  print(lf[i])
  d1 = read.csv(paste0("CSV/Subjects/",lf[i]))
  d = rbind(d, d1)
}

write.csv(d,"CSV/AllGazeTest.csv", row.names = F)
