debut <- Sys.time()

library(Epi)
library(dplyr)
library(corrplot)
library(leaps)
library(pscl)
library(ROCR)
library(descr)

# setwd("~/Memory")

#----------------------------------------

##########################
#                        #
#   CENSUS INCOME DATA   #
#                        #
##########################


#----------
# Loading data :
#----------

df <- read.csv("./cleaned_data.csv", sep=";", dec = ".")
test <- read.csv("./cleaned_test.csv", sep=";", dec = ".")

str(df)

df <- df %>% mutate(
  y_old = as.factor(y_old)
)

prop.table(table(df$y_old, useNA = "ifany"))*100
barplot(table(df$y_old))



#----------
# Check missing values :
#----------

pMiss <- function(x){sum(is.na(x))/length(x)*100}
sort(apply(df, 2, pMiss))
# Actually there are some missing values, but only on categorical features, and already encoded as '?'



#----------
# Check data distribution :
#----------

hist(df$fnlwgt)
boxplot(df$fnlwgt ~ df$y_old)

# !!! Takes time to run !!!
# scatterplotMatrix(~ education_num + fnlwgt + hours_per_week + capital_loss + capital_gain | y_old,
#                   data=train, smooth=T, reg.line=T, ellipse=T, by.groups=T)



#----------
# Check correlation / chisq :
#----------

X <- as.matrix(df[, (colnames(df) %in% c("fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"))])
X <- scale(X, center = T, scale = T)

corr <- cor(cbind(X), method = "pearson", use = "complete.obs")
corrplot(corr, order = "hclust", hclust.method = "ward.D2", diag = F, type="upper")

# Applying a correlation filter (no need here)
# highCorr <- findCorrelation(descrCorr, 0.85)
# X <- X[, -highCorr]
# X <- scale(X, center = T, scale = T)
# corr <- cor(X, method = "pearson", use = "complete.obs")
# corrplot(corr, order = "hclust")


# Sélectionner les variables qualitatives :
m = df[,c('education', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'native_country', 'income_bracket')]
success = df$y_old
# Initialisation de la table
table = data.frame(num_variable=integer(),nom_variable=character(),
                   chisq1=numeric(),chisq2=numeric(),chisq3=numeric())
for (i in seq(1, ncol(m))) {
  #Numéro itération
  num_variable = i
  nom_variable = colnames(m)[i]
  print(paste("Itération n°", i, "/",ncol(m), ":", nom_variable))
  #Proc Freq
  freq = table(m[,i], success, useNA='ifany')
  ch.test = chisq.test(freq)
  #Chisq
  chisq1 = ch.test$statistic
  chisq2 = ch.test$parameter
  chisq3 = ch.test$p.value
  #Ajout des statistiques dans une table
  ligne=cbind(num_variable,nom_variable,chisq1,chisq2,chisq3)
  table=rbind(table,ligne)
}

# On renomme les colonnes
names(table) = c("Num_variable","Nom_variable","Chi2","DF","Pvalue")
table <- table[order(table$Pvalue),]
rownames(table) <- NULL

print(table)



#----------
# MODEL 1 : Logistic regression (full model)
#----------

df$native_country[df$native_country == 'Holand-Netherlands'] = '?' # Too few obs
df$native_country = as.factor(as.character(df$native_country))
df$relationship_num = as.factor(df$relationship_num)

# Delete useless variables (info are in other features)
df_mod = df[, ! (colnames(df) %in% c('native_continent', 'workclass_num', 'over50K', 'marital_status_num', 'race_num', 'gender_num', 'education_num', 'relationship_num'))]
valid_id <- sample(nrow(df_mod), size = round(nrow(df)*0.25), replace = F)
train <- df_mod[-valid_id,]
valid <- df_mod[valid_id,]

# Use AIC & BIS to have a overview on the best features
stepwise <- regsubsets(y_old ~ . , data=train, method = "seqrep", nbest=1) # method = "forward" / "backward"
#par(mfrow=c(1,2))
plot(stepwise, scale = "adjr2", main = "Stepwise Selection\nAIC")
plot(stepwise, scale = "bic", main = "Stepwise Selection\nBIC")
#par(mfrow=c(1,1))

# Fit the model
model <- glm(y_old ~ . , family = binomial(link='logit'), data=train)

# Check model :
summary(model)
anova(model, test="Chisq")
print(pR2(model)[4])


# Quality on the validation set
pred <- predict(model, newdata = valid, type='response')
pred <- as.factor(ifelse(pred > 0.5,1,0))
misClasificError <- mean(pred != valid$y_old)
print(paste0('Accuracy = ', 1 - misClasificError)) #0.7163
p <- predict(model, newdata = valid, type="response")
pr <- prediction(p, valid$y_old)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
plot(prf, main = paste0("ROC curve\n(AUC = ", round(auc, 6), ")")) #0.7451
abline(a=0, b=1)


# Quality on the test set
test = test[,colnames(test) %in% colnames(valid)]
pred <- predict(model, newdata = test, type='response')
pred <- as.factor(ifelse(pred > 0.5,1,0))
misClasificError <- mean(pred != test$y_old)
print(paste0('Accuracy = ', 1 - misClasificError)) #0.7163
p <- predict(model, newdata = test, type="response")
pr <- prediction(p, test$y_old)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
auc <- performance(pr, measure = "auc")
auc <- auc@y.values[[1]]
plot(prf, main = paste0("ROC curve\n(AUC = ", round(auc, 6), ")")) #0.7451
abline(a=0, b=1)

#----------------------------------------
fin <- Sys.time()
print(fin-debut)


