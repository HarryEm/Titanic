# This is my first attempt at Kaggle and writing a full data project from start to 
# finish, hopefully this will be a clear demonstration of my thought process, any 
# feedback is more than welcome, especially if pointers for improvement in whichever
# aspect needs some work.

# Loading libraries
library(ggplot2) #charting
library(plyr) #data wrangling
library(dplyr) #data wrangling
library(Hmisc) #data wrangling
library(mice) #imputing variables
library(randomForest) #modelling
library(caret) #modelling

# Obtaining and reading in the data

if(!dir.exists("Kaggle/Titanic/")) {
        dir.create("Kaggle/Titanic/")}

if(!file.exists("./Kaggle/Titanic/train.csv")) {
        download.file("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/train.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1507864219&Signature=DDMnqFUFGMILVeFzSsfFM5plSpU50NdQTeFBD9c8Qzqgb02DzeLBDuZ7DKjND5MfyAFwcTI2jF43pS8geWc5Pecty%2FFALLZnS0r25Y5ly6hS37I1dKjSOjWMEQLZJ2tbmH0S2AxQfHqIzaGQ0%2F9GH2ujmHRaK5SEUPC%2BXFm0XouE4nr7VyXIeNW7vvsJ8yHdy4v3Y2omkJBTrhhiDG6mTEaW2SHTfYjn6qdnk%2FDrD9vVAlm2Bgq9B%2FOCUCk8E4OBtYpcTTnjXbFKnh6ZVeuAFBu%2FrdDOcnzN7VB6ZUiSdjl0vxt3fKhef3JQtt2UhXsPTqBu4a8%2FYtDMN0Wmos9QQQ%3D%3D","./Kaggle/Titanic/train.csv",method = "curl")}

if(!file.exists("./Kaggle/Titanic/test.csv")) {
        download.file("https://storage.googleapis.com/kaggle-competitions-data/kaggle/3136/test.csv?GoogleAccessId=competitions-data@kaggle-161607.iam.gserviceaccount.com&Expires=1507865074&Signature=c4sRIJ%2FwL5VFUXbvg%2BY84uH1hELjW9RA4RDhD%2FnlJfANBF%2BsQpAFUAcdlZRknYOadR91WBeZRf5d2iB4davXBQ5biGh8rt0%2FDQAtqQg1uXbJRd3n4BNFLB5zVmE5O7YQdIjYdwUz1FD2cdtmtFaDo%2BNzmDCcOv0HCUWhkt8jy8Z0blxvNGMdCmN%2Bn%2FFDUBvp3LJ7oHxC7vHeCBbmz5J6l%2F2UuGGIiR4ztQlCA58WXL6pX0pTSMGaYmqRnBoLL9RkqxxbkauERdekECcPyT4SoJq7Xq5wjmIiSkp8OiB8MBJcenVwmbwWhtbvDE1NBzhMLBD7DDqE1fUcf8Ea%2BXfYOA%3D%3D","./Kaggle/Titanic/test.csv",method = "curl")}

traindata <- read.csv("./Kaggle/Titanic/train.csv",stringsAsFactors = FALSE)
testdata <- read.csv("./Kaggle/Titanic/test.csv",stringsAsFactors = FALSE)

c(object.size(traindata),object.size(testdata)) ## data is small so performance shouldn't be an issue
#sum(traindata$PassengerId %in% testdata$PassengerId) == 0 ## check train and test set are disjoint

## I want to combine training and test sets because this makes it easier to perform the same 
## operations on both sets

testdata$Survived <- "NA"
merged <- rbind(traindata,testdata)

length(unique(merged$PassengerId)) == length(merged$PassengerId) ## check no duped entries

# Pre Clean Exploratory Data Analysis

head(merged) ## for detailed impormation about features check here https://www.kaggle.com/c/titanic/data

## I find it useful to guess what I think the data will look like, as this gives me a benchmark to 
## either support or attack in my initial analysis. Intuitively I would guess that anyone of higher
## class, either by ticket class itself or cost, would have a better chance of survival, as would
## women, children, and the elderly. Perhaps families too would be given a better chance of all getting
## into a lifeboat together as opposed to someone travelling alone?

colSums(is.na(merged))
colSums(merged=="")

## So there seems to be a lot of data missing in the Age variable and I would guess this contains a 
## lot of information. I have seen other people trying to impute age (use the mean for example) but 
## this is too much of a fudge for me and I'd rather train two models, one with age and one without.

## Let's look at charts of survivor trends by age, class, gender

merged$Pclass <- as.factor(merged$Pclass)
merged$Sex <- as.factor(merged$Sex)

g <- ggplot(merged[1:891,], aes(x=Pclass,fill=factor(Survived))) + geom_bar() + labs(fill = "Survived")
g <- g + labs(title="Survivor split by ticket class")
g

g <- g + facet_wrap(~Sex) + labs(title="Survivor split by ticket class and gender")
g

## As expected, higher class of ticket and female gender seem to be good positive predictors of survival.
## I would like to segment by age and create a new feature from age bracket as having individual ages in the 
## model would probably lead to overfitting.

qplot(merged$Age)

agebrackets <- c(0,13,18,30,55)
merged$Agebracket <- findInterval(merged$Age,agebrackets)

agetable <- data.frame(Agebracket=c(1,2,3,4,5),Age_range=c("<13","13-17","18-29","30-54","55+"))
merged <- join(merged,agetable,by="Agebracket")

g <- ggplot(merged[1:891,], aes(x=Age_range,fill=factor(Survived))) + geom_bar() + labs(fill = "Survived")
g <- g + labs(title="Survivor split by age group")
g

g <- g + facet_wrap(~Sex) + labs(title="Survivor split by age and gender")
g #make it go throught the x axis?

merged$Agebracket <- as.factor(merged$Agebracket)

## Age bracket seems to give a lot of information, with youger generally having a better chance
## of survival. Interestingly elderly women has a very good chance of survival whereas elderly men
## had a very bad chance so it looks like this is a useful division.


# Feature Engineering, Cleaning and Completing the Data

## As discussed before I intend to split up the data into those with Age data and those without, this 
## seems pretty central. However I want to see if this improves performance so I will do that once I've
## trained and tested a few models on the comibined set. The other variable with many missing entries
## is Cabin. I'm guessing this isn't just missing data, but that a lot of people didn't have a Cabin.
## So I'm thinking of turning this into a 2 factor variable, has / does not have cabin.

head(merged$Cabin,30)
length(unique(merged$Cabin))/length(merged$Cabin) ## only 14% are unique so there are a lot shared.
merged$Cabin[28] # this looks strange, multiple cabins on one ticket
subset(merged,Cabin == "C23 C25 C27") # it was one family, the Fortunes

## Thinking about adding a feature for how many people on the same ticket

merged$HasCabin <- as.factor(!(merged$Cabin==""))

g <- ggplot(merged[1:891,], aes(x=HasCabin,fill=factor(Survived))) + geom_bar()
g <- g +facet_wrap(~Pclass) + labs(title="Survivor split by class and Cabin")
g

## As expected few people in Class 2 and 3 had cabins, but actually those who did had a good chance of 
## survival. Helping to capture the smaller number who survived from lower classes should be very additive.

qplot(merged$Fare,bins=150)

## There doesn't seem to be natural brackets here unlike age, so I will just
## split in equal groups. There is one missing entry for Fare, I will impute
## the average Fare for his Pclass

subset(merged,is.na(merged$Fare))
merged[1044,]$Fare <- mean(subset(merged,Pclass==3)$Fare,na.rm=TRUE)

#Faretable <- count(merged,"Fare")
#
merged$Farebracket <- as.factor(cut2(merged$Fare,g=5))

g <- ggplot(merged[1:891,], aes(x=Farebracket,fill=factor(Survived))) + geom_bar()
g <- g +facet_wrap(~Pclass) + labs(title="Survivor split by class and Fare Bracket")
g <- g + theme(axis.text.x = element_text(angle = 90, hjust = 1))
g 

head(merged[order(merged$Fare,decreasing = FALSE),]$Fare)
subset(merged,Fare==0)

## This is more useful than expected, it does seem to split out the survivors within
## a class quite nicely

head(order(merged$Fare,decreasing = TRUE))
merged[259]

subset(merged,Fare==merged$Fare[259])

##One group paid over $500!

#is ticket overall cost or each ticket?

## Can we extract any information from the name?

## Just for fun: from https://en.wikipedia.org/wiki/Titanic_(1997_film) 
## Jack Dawson is actually Joseph Dawson, who is not in our sample,
## and neither is Rose. I didn't need a model to classify them anyway.

grep("Dawson",merged$Name)
grep("Rose",merged$Name)

merged$Title <- gsub('(.*, )|(\\..*)', '', merged$Name)

count(merged,Title)

VIP <- c("Capt","Col","Don","Dona","Dr","Jonkheer","Lady","Major",
         "Mlle", "Mme","Rev","Sir","the Countess")

merged$Title[merged$Title %in% VIP] <- "VIP"
merged[merged$Title=="Ms",]$Title <- "Mrs"

merged$Title <- as.factor(merged$Title)

count(merged,Title)

## I'm not that keen on only having 2 in the "Ms" camp
merged[merged$Title=="Ms",]$Title <- "Mrs"

g <- ggplot(merged[1:891,], aes(x=Title,fill=factor(Survived))) + geom_bar()
g <- g +facet_wrap(~Pclass) + labs(title="Survivor split by class and Title")
g

## I'm not sure how useful this is going to be based on the charts.
## Onto surname, I'm interested to see if there was some racial bias
## towards American / English

library(wru) # who r u - library to guess race from surname

merged$surname<- gsub("([A-Za-z]+).*", "\\1", merged$Name)

predict_race(merged,surname.only = TRUE)
raceprobs <- predict_race(merged,surname.only = TRUE)[18:22]
racepreds <- colnames(raceprobs)[apply(raceprobs,1,which.max)]
merged$Race <- as.factor(sub('.*\\.', '',racepreds))

g <- ggplot(merged[1:891,], aes(x=Race,fill=factor(Survived))) + geom_bar()
g <- g +facet_wrap(~Pclass) + labs(title="Survivor split by class and Race")
g

merged[order(merged$Fare,decreasing = TRUE),]

## Ticketsize variable
merged <- ddply(merged,.(Ticket),transform,Ticketsize=length(Ticket))
merged$Ticketsize <- as.factor(merged$Ticketsize)
merged <- merged[order(merged$PassengerId),] # ddply mixes up order

## Embarked
count(merged,Embarked) #I'm just going to use the most frequent here ie S
subset(merged,Embarked == "")
merged[c(62,830),"Embarked"] <- "S"

## To begin with, I want to use the mice library to impute the missing ages.
## Then I want to check if we get improvement by splitting data as explained
## above; this seems pivtotal to the analysis.

factors <- c("Pclass","Sex","Agebracket","Title")
set.seed(1234)
#m1 <- merged[, !names(merged) %in% c("Agebracket","Age_range")]
mice_ages <- mice(m1[, !names(m1) %in% factors], method='rf')
mice_out <- complete(mice_ages)

#mice_out$Agebracket <- findInterval(mice_out$Age,agebrackets)
#mice_out <- join(mice_out,agetable,by="Agebracket")

merged$Agebracket <- as.factor(merged$Agebracket)
merged$Embarked <- as.factor(merged$Embarked)


mergedages <- merged[,!names(merged) == "Age_range"]
mergedages$Age <- mice_out$Age

mergedages$Agebracket <- findInterval(mergedages$Age,agebrackets)
mergedages <- join(mergedages,agetable,by="Agebracket")

colSums(is.na(mergedages))
colSums(mergedages=="")


# Initial model fit

mergedagestrain <- mergedages[1:891,]
mergedagestest <- mergedages[892:1309,]
mergedagestrain$Survived <- as.factor(traindata$Survived)

set.seed(414)
inTrain<- createDataPartition(y=mergedagestrain$Survived,p=0.75, list=FALSE)
train <- mergedagestrain[inTrain,]
test <- mergedagestrain[-inTrain,]

rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Agebracket +
                                 Farebracket + Race + HasCabin + Ticketsize +
                                 Embarked + Title,
                         data = mergedagestrain,na.action = na.pass)

rf_model
rf_model$confusion
varImpPlot(rf_model)
importance(rf_model)

rf_model2 <- randomForest(factor(Survived) ~ Pclass + Sex + Agebracket +
                                 Farebracket + HasCabin + Ticketsize,
                         data = mergedagestrain,na.action = na.pass,ntree=2000)

rf_model2
varImpPlot(rf_model2)

rf_model3 <- randomForest(factor(Survived) ~ Pclass + Sex + Agebracket +
                                 Farebracket + Race + HasCabin + Ticketsize +
                                 Embarked,
                         data = mergedagestrain,na.action = na.pass,ntree=5000)

rf_model3
varImpPlot(rf_model3)

rf_model4 <- randomForest(factor(Survived) ~ Pclass + Sex +
                                 Farebracket + Ticketsize,
                         data = mergedagestrain,na.action = na.pass)

rf_model4
rf_model4$confusion

rf_model5 <- randomForest(factor(Survived) ~ Pclass + Sex +
                                Farebracket + Race + HasCabin + Ticketsize
                                  + Embarked,
                         data = mergedagestrain,na.action = na.pass)

rf_model5
rf_model5$confusion
plot(rf_model5)

rf_model6 <- randomForest(factor(Survived) ~ Pclass + Sex + Farebracket + HasCabin + Ticketsize + Embarked + Title,
                         data = mergedagestrain,na.action = na.pass,nodesize=20)

rf_model6 #looks the best
plot(rf_model6)
varImpPlot(rf_model6)
importance(rf_model6)

logreg <- glm(Survived ~ Pclass + Sex + Agebracket +
                      Farebracket + HasCabin + Ticketsize
               + Title + Embarked, family = binomial(link=logit), 
              data = train)

summary(logreg)
plot(logreg)

prediction <- predict(logreg,newdata=test,type="response")
prediction <- ifelse(prediction > 0.5,1,0)
misClasificError <- mean(prediction != test$Survived)
print(paste('Accuracy',1-misClasificError)) ## Accuracy 81%

## Lasso regularised logistic regression
library(glmnet)
x <- model.matrix(Survived ~ Pclass + Sex + Farebracket + HasCabin + Ticketsize + Embarked + Title,train)
cv.out <- cv.glmnet(x,y=train$Survived,alpha=1,family="binomial",type.measure = "mse") #select lambda -4

#best value of lambda
lambda_1se <- cv.out$lambda.1se

xtest <- model.matrix(Survived ~ Pclass + Sex + Farebracket + HasCabin + Ticketsize + Embarked + Title,test)
lasso_prob <- predict(cv.out,newx = xtest,s=lambda_1se,type="response")
#translate probabilities to predictions

lasso_predict <- rep("0",nrow(test))
lasso_predict[lasso_prob>.5] <-"1"
#confusion matrix
table(pred=lasso_predict,true=test$Survived)

mean(lasso_predict==test$Survived) #82% accuracy


## GBM

features <- c("Survived","Pclass","Sex","SibSp","Parch","HasCabin","Farebracket","Title","Ticketsize","Age_range")
fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)


gbm1 <- train(as.factor(Survived) ~ ., data = train[features], 
              method = "gbm", trControl = fitControl,verbose = FALSE)
prediction <- predict(gbm1, test[features],type= "prob")
prediction <- data.frame(ifelse(prediction[,2] > 0.5,1,0))
mean(prediction[,1] == test$Survived) #82% accuracy

fitControl2 <- trainControl(method = "repeatedcv", number = 6, repeats = 4)
gbm2 <- train(as.factor(Survived) ~ ., data = train[features], 
              method = "gbm", trControl = fitControl,verbose = FALSE)
prediction <- predict(gbm2, test[features],type= "prob")
prediction <- data.frame(ifelse(prediction[,2] > 0.5,1,0))
mean(prediction[,1] == test$Survived) #82.5% accuracy

gbm3 <- train(as.factor(Survived) ~ ., data = rbind(train[features],test[features]), 
              method = "gbm", trControl = fitControl,verbose = FALSE)
#use test data as well to train the model - did not improve

gbm4 <- gbm(as.factor(Survived)~.,data=train[features],distribution="bernoulli",n.trees=5000,interaction.depth=4)
prediction <- predict(gbm4, newdata=test[features],n.trees=5000,type="response")
prediction <- data.frame(ifelse(prediction[,2] > 0.5,1,0))
mean(prediction[,1] == test$Survived)


## Xgboost
library(xgboost) #modelling

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv(params = params, data = train[features], nrounds = 100, nfold = 5, 
                showsd = T, stratified = T, print.every.n = 10, early.stop.round = 20, maximize = F)

## Now onto training two models depending on whether Age is available, need to split up data

row.names(merged) <- merged$PassengerId
mergedtrain <- merged[1:891,]
mergedtest <- merged[892:1309,]
mergedtrain$Survived <- as.factor(mergedtrain$Survived)

sum(is.na(mergedtrain$Age))
sum(is.na(mergedtest$Age))



mergedtrain[is.na(mergedtrain$Age),]
mergedtrain[!is.na(mergedtrain$Age),]


rf_model_ages <- randomForest(factor(Survived) ~ Pclass + Sex + Farebracket + HasCabin + Agebracket + Ticketsize + Embarked + Title,
                              data = mergedtrain[!is.na(mergedtrain$Age),],nodesize=20)

rf_model_ages
rf_model_ages$confusion
varImpPlot(rf_model_ages)
importance(rf_model_ages)


rf_model_noages <- randomForest(factor(Survived) ~ Pclass + Sex +
                                      Farebracket + Race + HasCabin + Ticketsize +
                                      Embarked + Title,
                              data = mergedtrain[is.na(mergedtrain$Age),],nodesize=20)

rf_model_noages
rf_model_noages$confusion
varImpPlot(rf_model_noages)
importance(rf_model_noages)

p1 <- predict(rf_model_ages, mergedtest[!is.na(mergedtest$Age),])
p2 <- predict(rf_model_noages, mergedtest[is.na(mergedtest$Age),])
prediction <- join(data.frame(PassengerId=names(p1),Survived=p1),data.frame(PassengerId=names(p2),Survived=p2))

# This actually didn't help my position on the leaderboard, pretty surprised by this


prediction <- predict(rf_model, mergedagestest)

prediction <- predict(logreg,newdata=mergedagestest,type="response")
prediction <- ifelse(prediction > 0.5,1,0)

mergedagestest[mergedagestest$Title=="Ms",]$Title <- "Mrs"
prediction <- predict(gbm1, mergedagestest[features],type= "prob")
prediction <- data.frame("PassengerID" = mergedagestest$PassengerId,"Survived"=ifelse(prediction[,2] > 0.5,1,0))

prediction <- predict(gbm2, mergedagestest[features],type= "prob")
prediction <- data.frame("PassengerID" = mergedagestest$PassengerId,"Survived"=ifelse(prediction[,2] > 0.5,1,0))

prediction <- predict(gbm3, mergedagestest[features],type= "prob")
prediction <- data.frame("PassengerID" = mergedagestest$PassengerId,"Survived"=ifelse(prediction[,2] > 0.5,1,0))


submission <- data.frame(PassengerId=names(prediction),Survived=prediction)
write.csv(submission, file = "./Kaggle/Titanic/predictions.csv",row.names = F)

if(!file.exists("./Kaggle/Titanic/predictions.csv")) {
        write.csv(submission, file = "./Kaggle/Titanic/predictions.csv",row.names = F)}
