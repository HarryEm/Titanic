library(ggplot2) #charting
library(scales) #charting
library(grid) #charting
library(plyr) #data wrangling
library(dplyr) #data wrangling
library(Hmisc) #data wrangling
library(mice) #imputing variables
library(randomForest) #modelling
library(caret) #modelling

traindata <- read.csv('./Kaggle/input/train.csv', stringsAsFactors = F)
testdata <- read.csv('./Kaggle/input/test.csv', stringsAsFactors = F)

testdata$Survived <- "NA"
merged <- rbind(traindata,testdata)

a <- colSums(is.na(testdata))+colSums(testdata=="")
a <- names(a[is.na(a)|a!=0])
a

missing <- c()

for (i in a) {
  missing <- paste(missing,as.integer(!is.na(merged[i])^!merged[i]==""),sep="")
}

merged[missing=="100",]

table(missing)
merged$Missing <- missing

merged$Pclass <- as.factor(merged$Pclass)
merged$Sex <- as.factor(merged$Sex)

agebrackets <- c(0,13,18,30,55)
merged$Agebracket <- findInterval(merged$Age,agebrackets)

agetable <- data.frame(Agebracket=c(1,2,3,4,5),Age_range=c("<13","13-17","18-29","30-54","55+"))
merged <- join(merged,agetable,by="Agebracket")
merged$Agebracket <- as.factor(merged$Agebracket)

merged$HasCabin <- as.factor(!(merged$Cabin==""))

a <- subset(merged,is.na(merged$Fare))
a
merged[a[,1],]$Fare <- mean(subset(merged,Pclass==3)$Fare,na.rm=TRUE)

merged$Farebracket <- as.factor(cut2(merged$Fare,g=5))


merged$Title <- gsub('(.*, )|(\\..*)', '', merged$Name)

count(merged,Title)

merged$Title <- as.factor(merged$Title)


merged <- ddply(merged,.(Ticket),transform,Ticketsize=length(Ticket))
merged$Ticketsize <- as.factor(merged$Ticketsize)
merged <- merged[order(merged$PassengerId),] # ddply mixes up order

subset(merged,Embarked == "")
merged[c(62,830),"Embarked"] <- "S"
merged$Embarked <- as.factor(merged$Embarked)

m1 <- merged[, !names(merged) %in% c("Agebracket","Age_range")]
mice_ages <- mice(m1[, !names(m1) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived','Missing')], method='rf',seed = 1234)
mice_out <- mice::complete(mice_ages)

merged$Age <- mice_out$Age
merged$Agebracket <- findInterval(merged$Age,agebrackets)
merged <- join(merged,agetable,by="Agebracket")

colSums(is.na(merged))+colSums(merged=="")

mergedtrain <- merged[1:891,]
mergedtest <- merged[892:1309,]
mergedtrain$Survived <- as.factor(traindata$Survived)

set.seed(414)
inTrain<- createDataPartition(y=mergedtrain$Survived,p=0.75, list=FALSE)
train <- mergedtrain[inTrain,]
test <- mergedtrain[-inTrain,]

dftemp <- mergedtrain[,c("Survived","Pclass","Sex","Farebracket","Age","HasCabin","Ticketsize","Embarked","Title","Missing")]

