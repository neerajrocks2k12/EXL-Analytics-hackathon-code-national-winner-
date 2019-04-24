library(ROCR)
library(randomForest)
library(corrplot)
library(Hmisc)
library(MASS)
library(caret)

setwd('G:/exl/processeddata')
health_train <- read.csv('health_train.csv')
cont_var <- c('Demo2','Demo3','Demo4','Demo6','DisHis1Times','DisHis2Times','DisHis3Times','DisHis6','LungFun1','LungFun2','LungFun3','LungFun4','LungFun5','LungFun6','LungFun7','LungFun8','LungFun9','LungFun10','LungFun11','LungFun12','LungFun13','LungFun14','LungFun15','LungFun16','LungFun17','LungFun18','LungFun20','Dis2Times','Dis3Times','RespQues1','ResQues1a','ResQues1b','ResQues1c','ResQues2a','SmokHis1','SmokHis2','SmokHis3','SmokHis4')
cont_var_df <- data.frame(health_train[, cont_var])

pear_corr_mat <- cor(cont_var_df, method='pearson')

png(height=768, width=1500, pointsize=25, file='corrplot.png')
corrplot(pear_corr_mat, method='circle',tl.cex=0.7,mar=c(0,0,1,0),title='Pearson\'s correlation coefficient matrix')
dev.off()

write.matrix(pear_corr_mat,file='corr_mat.csv',sep=',')

#===============================================================#
summary(health_train)
dummydf <- TestData#health_train

#converting to factor variables
dummydf$DisStage1 <- as.factor(dummydf$DisStage1)
dummydf$DisStage2 <- as.factor(dummydf$DisStage2)
dummydf$Dis1 <- as.factor(dummydf$Dis1)
dummydf$Dis2 <- as.factor(dummydf$Dis2)
dummydf$Dis3 <- as.factor(dummydf$Dis3)
dummydf$Dis4 <- as.factor(dummydf$Dis4)
dummydf$Dis5 <- as.factor(dummydf$Dis5)
dummydf$Dis6 <- as.factor(dummydf$Dis6)
dummydf$EXAC <- as.factor(dummydf$EXAC)
dummydf$Demo1 <- as.factor(dummydf$Demo1)
dummydf$Demo5 <- as.factor(dummydf$Demo5)
dummydf$DisHis1 <- as.factor(dummydf$DisHis1)
dummydf$DisHis2 <- as.factor(dummydf$DisHis2)
dummydf$DisHis3 <- as.factor(dummydf$DisHis3)
dummydf$DisHis4 <- as.factor(dummydf$DisHis4)
dummydf$DisHis5 <- as.factor(dummydf$DisHis5)
dummydf$DisHis7 <- as.factor(dummydf$DisHis7)
dummydf$LungFun19 <- as.factor(dummydf$LungFun19)
dummydf$Dis1Treat <- as.factor(dummydf$Dis1Treat)
dummydf$Dis4Treat <- as.factor(dummydf$Dis4Treat)
dummydf$Dis5Treat <- as.factor(dummydf$Dis5Treat)
dummydf$Dis6Treat <- as.factor(dummydf$Dis6Treat)
dummydf$Dis7 <- as.factor(dummydf$Dis7)

set.seed(42)
dummydf[, 'train']<- ifelse(runif(nrow(dummydf))<0.80,1,0)
trainColnum <- grep('train', names(dummydf))

train <- dummydf[dummydf$train==1, -trainColnum]
test <- dummydf[dummydf$train==0, -trainColnum]

#bestmtry <- tuneRF(dummydf[-1], dummydf$EXAC, ntreeTry=100,stepFactor=2,improve=0.01,trace=T,plot=T,dobest=F)
rf <- randomForest(EXAC~., train[-1],mtry=8,ntree=1000,nodesize=3,replace=F, keep.forest=T,importance=T,strata=train$EXAC,sampsize=c('0'=200,'1'=125))
rf
#importance(rf)
varImpPlot(rf)
pred <- predict(rf, test[-1])
xtab <- table(pred=pred, true=test$EXAC)
confusionMatrix(xtab)

pr <- prediction(pred, test$EXAC)

pairs(cont_var_df[,c(1,2,3,4,5,6)])
x11()
pairs(cont_var_df[,c(7,8,9,10,11,12)])
pairs(cont_var_df[,c(16:22)])

#==================================#
#caret wala rf
ctrl <- trainControl(method='cv', repeats=5)
train$EXAC <- gsub('0','r',train$EXAC)
train$EXAC <- gsub('1','s',train$EXAC)
rf_caret <- train(EXAC~.,train[-1],method='rf',ntree=1000,tuneLength=10,trControl=ctrl,allowParallel=T,strata=as.factor(train$EXAC),sampsize=c('r'=150,'s'=130))
rf2
rf2$finalModel
xtab2 <- predict(rf_caret, test)
test$EXAC <- gsub('0','r',test$EXAC)
test$EXAC <- gsub('1','s',test$EXAC)
confusionMatrix(table(pred=xtab2,true=test$EXAC))

varImp(rf_caret)#caret varimp
png(height=768, width=1500, pointsize=30, file='varImpplot_caret.png')
plot(varImp(rf_caret))
dev.off()

# rfecontrol <- rfeControl(functions=rfFuncs, method="cv", repeats=5, verbose=FALSE,number=5)
# results <- rfe(train, train$EXAC, rfeControl=rfecontrol,metric='Accuracy')
# x11()
# plot(results)

#=====saving model====#
save(rf2,file='rf2.RData')
save(train, file='train.RData')
save(test, file='test.RData')
load('rf2.RData')

#PREDictions without PCA
TestPredict <- predict(rf_caret, dummydf, type='prob')
TestPredict
sum(TestPredict==0)
write.csv(data.frame(predict=TestPredict), file='predictions_caret.csv')
