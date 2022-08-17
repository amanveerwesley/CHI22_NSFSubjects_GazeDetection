library(tidyverse)
library(caret)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

d = read.csv("Data/CM.csv")

fk = read.csv("Data/UniformRandomDistribution_2000_fk.csv")
fk = fk[1:500,]

d["Fettah"] = fk$Fettah
d["FettahNotes"] = fk$FettahNotes


a = confusionMatrix(data = factor(d$Aman), reference = factor(d$AlgoOutput2))
f = confusionMatrix(data = factor(d$Fettah), reference = factor(d$AlgoOutput2))


#Aman
al = filter(d, AlgoOutput2 == "L")
ac = filter(d, AlgoOutput2 == "C")
ar = filter(d, AlgoOutput2 == "R")
ax = filter(d, AlgoOutput2 == "X")

alcm = confusionMatrix(reference = factor(al$Aman), data = factor(al$AlgoOutput2))
accm = confusionMatrix(reference = factor(ac$Aman), data = factor(ac$AlgoOutput2))
arcm = confusionMatrix(reference = factor(ar$Aman), data = factor(ar$AlgoOutput2))
axcm = confusionMatrix(reference = factor(ax$Aman), data = factor(ax$AlgoOutput2))


#Fettah
fl = filter(d, AlgoOutput2 == "L")
fc = filter(d, AlgoOutput2 == "C")
fr = filter(d, AlgoOutput2 == "R")
fx = filter(d, AlgoOutput2 == "X")

flcm = confusionMatrix(reference = factor(fl$Fettah), data = factor(fl$AlgoOutput2))
fccm = confusionMatrix(reference = factor(fc$Fettah), data = factor(fc$AlgoOutput2))
frcm = confusionMatrix(reference = factor(fr$Fettah), data = factor(fr$AlgoOutput2))
fxcm = confusionMatrix(reference = factor(fx$Fettah), data = factor(fx$AlgoOutput2))





d1 = d
d1["Agreement"] = 0
d1 = d1[-which(is.na(d1$Aman)),]
for (j in 1:nrow(d1)) {
  print(j)
  if (d1$Aman[j] == d1$Fettah[j]) {
    d1$Agreement[j] = 1
  }
}



