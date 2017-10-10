# Loading libraries
library(ggplot2) #charting
library(plyr) #data wrangling

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

## I want to combine training and test sets because this makes it easier to perform same 
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

merged$Survived <- as.factor(merged$Survived)
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

## There are a lot of Fares around 10 so these should have 

Faretable <- count(merged,"Fare")
