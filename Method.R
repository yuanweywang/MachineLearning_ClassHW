# 下載套件
library(caTools) # 切train&test的工具
library(randomForest) # 資料差捕、random forest
library(e1071) #SVM
library(ggplot2) #caret需要
library(lattice) #caret需要
library (caret) #做dummy
library(xgboost) #XGBoost
library(neuralnet) #NN

# 載入資料
pokemon <- read.csv("/Users/oumotoutsu/Desktop/Github/MachineLearning_ClassHW/pokemon_data.csv",sep=",",header=TRUE)
head(pokemon)

# 補NA、變成factor
pokemon[pokemon == ""]<-NA
str(pokemon)
pokemon$Type_1 <- as.factor(pokemon$Type_1)
pokemon$Type_2 <- as.factor(pokemon$Type_2)
pokemon$Generation <- as.factor(pokemon$Generation)
pokemon$Legendary <- as.factor(pokemon$Legendary)  
str(pokemon)

# 資料差捕、切割資料training & testing、處理會用到的變數
pokemon1 <- subset(pokemon, select=c(-TOTAL, -ID, -Name))
set.seed(42)
impute1 <- rfImpute(Legendary ~ ., data = pokemon1, iter=5)
pokemon2 <- subset(pokemon, select=c(-Legendary, -ID, -Name))
set.seed(43)
impute2 <- rfImpute(TOTAL ~ ., data = pokemon2, iter=5)

set.seed(100)
split1 <- sample.split(impute1$Type_1, SplitRatio = 0.7)
train1 <- subset(impute1,split1==TRUE)      #train1
test1  <- subset(impute1,split1==FALSE)
x_test1<- subset(test1, select=-Legendary)  #x_test1
y_test1<-test1$Legendary                    #y_test1

set.seed(101)
split2 <- sample.split(impute2$Type_1, SplitRatio = 0.7)
train2 <- subset(impute2,split2==TRUE)      #train2
test2  <- subset(impute2,split2==FALSE)     
x_test2<- subset(test2, select=-TOTAL)      #x_test2 
y_test2<-test2$TOTAL                        #y_test2

x_train1<-subset(train1, select=-Legendary) #x_train1
y_train1<-train1$Legendary                  #y_train1

x_train2<-subset(train2, select=-TOTAL)     #x_train2
y_train2<-train2$TOTAL                      #y_train2

xtrain1dummy<- dummyVars(" ~ . ", data=x_train1)
x_train1_dummy <- data.frame(predict(xtrain1dummy, newdata=x_train1)) #x_train1_dummy

x_test1_dummy <- dummyVars(" ~.  ", data=x_test1)
x_test1_dummy<- data.frame(predict(x_test1_dummy, newdata=x_test1))   #x_test1_dummy

x_train2_dummy <- dummyVars(" ~.  ", data=x_train2)     
x_train2_dummy<- data.frame(predict(x_train2_dummy, newdata=x_train2))#x_train2_dummy

x_test2_dummy <- dummyVars(" ~.  ", data=x_test2)
x_test2_dummy<- data.frame(predict(x_test2_dummy, newdata=x_test2))   #x_test2_dummy

train2_dummy <- dummyVars("  ~.  ", data=train2)
train2_dummy <- data.frame(predict(train2_dummy , newdata=train2))    #train2_dummy

# 1-1 SVM 類別(Legendary, acc=90.83%)----
# 沒有fine-tune (acc=95.42%)
svm_model1 <- svm(Legendary~.,data=train1)
pred1_1_1<-predict(svm_model1,x_test1)
table1_1_1<-table(y_test1,pred1_1_1)
print(table1_1_1)
accuracy<-(table1_1_1[1,1]+table1_1_1[2,2])/(length(y_test1))
print(accuracy) 

# 經過fine-tune (acc=95.42%)
svm_tune<-tune(svm, train.x=x_train1_dummy, train.y=y_train1, kernal='radial', ranges=list(cost=10^(-1:2), gamma=c(.5,2,3)))
print(svm_tune)
svm_model2 <- svm(Legendary~.,data=train1, kernel='radial', cost=10, gamma=0.5)
pred1_1_2<-predict(svm_model2,x_test1)
table1_1_2<-table(y_test1,pred1_1_2)
print(table1_1_2)
accuracy<-(table1_1_2[1,1]+table1_1_2[2,2])/(length(y_test1))
print(accuracy) 

# 1-2 Logistic Regression 類別(Legendary, acc=95.42%))----
train1_ynum <- train1
train1_ynum$Legendary <- as.numeric(train1_ynum$Legendary) - 1  

logistic_model <- glm(Legendary ~ ., data=train1_ynum, family="binomial")
pred1_2 <- predict(logistic_model, x_test1, type='response')
pred1_2 <- ifelse(pred1_2 > 0.5,1,0)
table1_2<-table(y_test1,pred1_2)
print(table1_2)
accuracy<-(table1_2[1,1]+table1_2[2,2])/(length(y_test1))
print(accuracy)

# 1-3 Random Forest 類別(Legendary, acc=95.42%))----
set.seed(70)
RF_model1 <- randomForest(Legendary ~ ., data=train1, proximity=TRUE)
pred1_3<-predict(RF_model1,x_test1)
table1_3<-table(y_test1,pred1_3)
print(table1_3)
accuracy<-(table1_3[1,1]+table1_3[2,2])/(length(y_test1))
print(accuracy)

# 1-4 XGBoost 類別(Legendary, acc=95.83%))----
y_train1_num <- as.numeric(train1$Legendary) - 1                    
xgb_model <- xgboost(data = as.matrix(x_train1_dummy),label=y_train1_num, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
xgb_model
pred1_4 <- predict(xgb_model, as.matrix(x_test1_dummy))
pred1_4 <- ifelse(pred1_4 > 0.5,"TRUE","FALSE")
table1_4<-table(y_test1,pred1_4)
print(table1_4)
accuracy<-(table1_4[1,1]+table1_4[2,2])/(length(y_test1))
print(accuracy)

# 1-5 Neural Networks 類別(Legendary, acc=95.42%))----
train1_nn<-cbind(subset(train1, select=Legendary), x_train1_dummy)
nn_model1 <- neuralnet(Legendary~., data=train1_nn, hidden=50, act.fct='logistic', linear.output = FALSE)

pred1_5 <- predict(nn_model1,x_test1_dummy)
pred1_5 <- round(pred1_5)
pred1_5 <- as.data.frame(pred1_5)
pred1_5$Legendary <- ""
for(i in 1:nrow(pred1_5)){
  if(pred1_5[i, 1]==1){ pred1_5[i, "Legendary"] <- "FALSE"}
  if(pred1_5[i, 2]==1){ pred1_5[i, "Legendary"] <- "TRUE"}}
table1_5<-table(y_test1,pred1_5$Legendary)
print(table1_5)
accuracy<-(table1_5[1,1]+table1_5[2,2])/(length(y_test1))
print(accuracy)

# 2-1 Random Forest 連續(TOTAL, mse=659.8309)----
set.seed(70)
RF_model2 <- randomForest(TOTAL ~ ., data=train2, proximity=TRUE, mtry = 13, importance = TRUE,ntree=2000)
RF_model2
pred2_1<-predict(RF_model2,x_test2)
print(mean((y_test2-pred2_1)^2))

# 2-2 XGBoost 連續(TOTAL, mse=1650.683)----
xgb_model2 <- xgboost(data = as.matrix(x_train2_dummy),label=y_train2, max.depth = 2, eta = 1, nthread = 2, nrounds = 2)
xgb_model2
pred2_2 <- predict(xgb_model2, as.matrix(x_test2_dummy))
mean((y_test2-pred2_2)^2)

# 2-3 Neural Networks 連續(TOTAL, mse=27631.82)----
train2_dummy_scaled = scale(train2_dummy)
train2_dummy_scaled = data.frame(train2_dummy_scaled)

nn_model2 <- neuralnet(TOTAL~., data=train2_dummy_scaled, hidden=50, act.fct = sigmoid,err.fct = "sse", linear.output = T)

pred2_3<-predict(nn_model2,x_test2_dummy)
mean((y_test2-as.vector(pred2_3))^2)

