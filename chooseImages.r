library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

d = read.csv("CSV/UniformRandomDistribution_2000.csv")

for (i in 1:nrow(d)) {
  d1 = d[i,]
  sub = d1$Participant_ID
  second = d1$Seconds * 10
  fileAddress = paste0('Frames/',sub,"/",second,".jpg") 
  newAddress = paste0('Frames/Frames2000/')
  file.copy(from = fileAddress, to = newAddress)
  file.rename(from = paste0('Frames/Frames2000/',second,'.jpg'), 
              to = paste0('Frames/Frames2000/',d$RandomKey,'.jpg'))
  
}