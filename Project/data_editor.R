rm(list=ls())
df <- read.table(file="C:/Users/Mahin/Google Drive/Workspace/Local Accuracy of Classifiers/Project/Data Set Credit_g/credit.data",header=F, sep="")
write.table(df, file="C:/Users/Mahin/Google Drive/Workspace/Local Accuracy of Classifiers/Project/Data Set Credit_g/credit.data", row.names=FALSE, col.names=FALSE, sep=",")
