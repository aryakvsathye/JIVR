library(dplyr)
library(rugarch)
beta = read.csv('implied_vol_model.csv')
dates = as.Date(beta$as_of)
T = length(dates)
beta_1 <- data.matrix(beta[, 'beta_1'])
beta_2 <- data.matrix(beta[, 'beta_2'])
beta_3 <- data.matrix(beta[, 'beta_3'])
beta_4 <- data.matrix(beta[, 'beta_4'])
beta_5 <- data.matrix(beta[, 'beta_5'])
rm(beta)

db = read.csv('db.csv')
db$as_of = as.Date(db$as_of)
y = db[, c('as_of', 'as_of_stock_price')]
rm(db)
y <- unique(y)
y <- y %>% filter(as_of >= '2022-12-28') %>% arrange(as_of) %>% rename(y = as_of_stock_price)
y <- log(y[, 'y'])
y <- data.matrix(diff(y))

r_q = read.csv('r_q_df.csv')
r_q$as_of = as.Date(r_q$as_of)
r_q <- r_q %>% filter(as_of >= '2022-12-28') %>% arrange(as_of)
r <- data.matrix(r_q[, 'r'])
q <- data.matrix(r_q[, 'q'])
rm(r_q)

ATM_1mo_IV = data.matrix(beta_1 + beta_2*exp(-sqrt(1/3)))^2
# Model for Y
variance_y = list(garchOrder=c(0, 0), external.regressors=data.matrix(ATM_1mo_IV[1:T-1]))
mean_y = list(armaOrder=c(0, 0), include.mean=FALSE, external.regressors=data.matrix(r[1:T-1] - q[1:T-1]))
spec = ugarchspec(variance.model=variance_y,
                  mean.model=mean_y,
                  distribution.model="nig")
model_y = ugarchfit(data=y, spec=spec, solver='hybrid')
model_y

# Model for beta_1
variance_beta_1 = list(model='fGARCH', garchOrder=c(1, 0), submodel='NAGARCH')
mean_beta_1 = list(armaOrder=c(1, 0), include.mean=FALSE, external.regressors=cbind(beta_2[1:T-1]))
spec = ugarchspec(variance.model=variance_beta_1,
                  mean.model=mean_beta_1,
                  distribution.model="nig",
                  start.pars=list(eta21=-0.1118))
model_beta_1 = ugarchfit(data=data.matrix(beta_1[2:T]), spec=spec, solver='hybrid')
model_beta_1

# Model for beta_2
variance_beta_2 = list(model='fGARCH', garchOrder=c(1, 0), submodel='NAGARCH')
mean_beta_2 = list(armaOrder=c(2, 0), include.mean=TRUE, external.regressors=cbind(beta_3[1:T-1], beta_5[1:T-1]))
spec = ugarchspec(variance.model=variance_beta_2,
                  mean.model=mean_beta_2,
                  distribution.model="nig",
                  start.pars=list(eta21=-3.9178))
model_beta_2 = ugarchfit(data=data.matrix(beta_2[2:T]), spec=spec, solver='hybrid')
model_beta_2

# Model for beta_3
variance_beta_3 = list(model='fGARCH', garchOrder=c(1, 0), submodel='NAGARCH')
mean_beta_3 = list(armaOrder=c(1, 0), include.mean=TRUE, external.regressors=cbind(beta_2[1:T-1], beta_4[1:T-1], beta_5[1:T-1]))
spec = ugarchspec(variance.model=variance_beta_3,
                  mean.model=mean_beta_3,
                  distribution.model="nig",
                  start.pars=list(eta21=0.1935))
model_beta_3 = ugarchfit(data=data.matrix(beta_3[2:T]), spec=spec, solver='hybrid')
model_beta_3

# Model for beta_4
variance_beta_4 = list(model='fGARCH', garchOrder=c(1, 0), submodel='NAGARCH')
mean_beta_4 = list(armaOrder=c(1, 0), include.mean=TRUE, external.regressors=cbind(beta_3[1:T-1]))
spec = ugarchspec(variance.model=variance_beta_4,
                  mean.model=mean_beta_4,
                  distribution.model="nig",
                  start.pars=list(eta21=0.1211))
model_beta_4 = ugarchfit(data=data.matrix(beta_4[2:T]), spec=spec, solver='hybrid')
model_beta_4

# Model for beta_5
variance_beta_5 = list(model='fGARCH', garchOrder=c(1, 0), submodel='NAGARCH', external.regressors=data.matrix(rep(var(beta_5), T-1)))
mean_beta_5 = list(armaOrder=c(2, 0), include.mean=TRUE, external.regressors=cbind(beta_1[1:T-1], beta_4[1:T-1]))
spec = ugarchspec(variance.model=variance_beta_5,
                  mean.model=mean_beta_5,
                  distribution.model="nig",
                  start.pars=list(eta21=-0.2060))
model_beta_5 = ugarchfit(data=data.matrix(beta_5[2:T]), spec=spec, solver='hybrid')
model_beta_5

library(jsonify)
y_theta = list()
for(name in names(coef(model_y))) {
  y_theta[name] = coef(model_y)[name]
}
beta_1_theta = list()
for(name in names(coef(model_beta_1))) {
  beta_1_theta[name] = coef(model_beta_1)[name]
}
beta_2_theta = list()
for(name in names(coef(model_beta_2))) {
  beta_2_theta[name] = coef(model_beta_2)[name]
}
beta_3_theta = list()
for(name in names(coef(model_beta_3))) {
  beta_3_theta[name] = coef(model_beta_3)[name]
}
beta_4_theta = list()
for(name in names(coef(model_beta_4))) {
  beta_4_theta[name] = coef(model_beta_4)[name]
}
beta_5_theta = list()
for(name in names(coef(model_beta_5))) {
  beta_5_theta[name] = coef(model_beta_5)[name]
}
params = list(y=y_theta, beta_1=beta_1_theta, beta_2=beta_2_theta, beta_3=beta_3_theta, beta_4=beta_4_theta, beta_5=beta_5_theta)
fileConn<-file("saved_model/ts_model.json")
writeLines(to_json(params), fileConn)
close(fileConn)

y_residuals <- residuals(model_y, standardize=TRUE)
beta_1_residuals <- residuals(model_beta_1, standardize=TRUE)
beta_2_residuals <- residuals(model_beta_2, standardize=TRUE)
beta_3_residuals <- residuals(model_beta_3, standardize=TRUE)
beta_4_residuals <- residuals(model_beta_4, standardize=TRUE)
beta_5_residuals <- residuals(model_beta_5, standardize=TRUE)
residuals <- as.matrix(cbind(y_residuals, beta_1_residuals, beta_2_residuals, beta_3_residuals, beta_4_residuals, beta_5_residuals))
rownames(residuals) <- NULL
write.csv(residuals, 'saved_model/residuals.csv', row.names=FALSE)
h_y <- sigma(model_y)
h_beta_1 <- sigma(model_beta_1)
h_beta_2 <- sigma(model_beta_2)
h_beta_3 <- sigma(model_beta_3)
h_beta_4 <- sigma(model_beta_4)
h_beta_5 <- sigma(model_beta_5)
h <- as.matrix(cbind(h_y, h_beta_1, h_beta_2, h_beta_3, h_beta_4, h_beta_5))
rownames(h) <- NULL
write.csv(h, 'saved_model/conditional_var.csv', row.names=FALSE)
