clc, clear all, close all
rng('default');
homedata = readtable("home_data.csv"); %Read data
%select attribute
homedata1 = table2array(homedata(:,[1:2,5:7,22]));
Y =homedata1(:,6);
X =homedata1(:,1:5);
%Data Visualize
varname = {'numbed','yearbuilt', 'numroom', 'numbath', 'livingarea', 'price'};
%scatter data 
figure
for i= 1:5
    subplot(3,2,i);
    scatter(X(:,i),Y);
    xlabel(varname(i));
    ylabel(varname(6)); 
end

corrplot(homedata1,'varNames', varname) %correlation matrix
figure
histfit(Y) %histogram with distribution fit
figure
heatmap(varname,varname,corr(homedata1)) %Relationship between attributes
figure
%Cross varidation for X (train: 80%, test: 20%)
cv_x = cvpartition(size(X,1),'HoldOut',0.2);
%Separate to training and test data
X_Train = X(cv_x.training,:);
X_Test  = X(cv_x.test,:);
%Cross varidation for Y (train: 80%, test: 20%)
cv_y = cvpartition(size(Y,1),'HoldOut',0.2);
%Separate to training and test data
Y_Train = Y(cv_y.training,:);
Y_Test  = Y(cv_y.test,:);
%Linear regression
mdl =fitlm(X,Y,'linear');
%Linear Regression Train
Y_pred_Train = mdl.predict(X_Train);
R2_Train = rsquare(Y_pred_Train, Y_Train)
RMSE_Train =sqrt(mse(Y_Train, Y_pred_Train))
scatterplot(X_Train, Y_Train, Y_pred_Train,varname)
figure

%Linear Regression Test
Y_pred_Test = mdl.predict(X_Test);
R2_Test = rsquare(Y_pred_Test, Y_Test)
RMSE_Test =sqrt(mse(Y_Test, Y_pred_Test))
scatterplot(X_Test, Y_Test, Y_pred_Test,varname)
figure

%Lasso_Train
[B,FitInfo] = lasso(X,Y,'lambda',2);
predicted_train_values = X_Train*B+FitInfo.Intercept;
R2_Train_Lasso = rsquare(predicted_train_values, Y_Train)
RMSE_Train_Lasso =sqrt(mse(Y_Train, predicted_train_values))
scatterplot(X_Train, Y_Train, predicted_train_values,varname)
figure

%Lasso_Test
predicted_test_values = X_Test*B+FitInfo.Intercept;
R2_Test_Lasso = rsquare(predicted_test_values, Y_Test)
RMSE_Test_Lasso =sqrt(mse(Y_Test, predicted_test_values))
scatterplot(X_Test, Y_Test, predicted_test_values,varname)
figure

%Ridge_Train
b = ridge(Y,X,2,0);
y_train_predicted = b(1) + X_Train*b(2:end);
R2_Train_Ridge = rsquare(y_train_predicted, Y_Train)
RMSE_Train_Ridge =sqrt(mse(Y_Train, y_train_predicted))
scatterplot(X_Train,Y_Train, y_train_predicted, varname)
figure

%Ridge_Test
y_test_predicted = b(1) + X_Test*b(2:end);
R2_Test_Ridge = rsquare(y_test_predicted, Y_Test)
RMSE_Test_Ridge =sqrt(mse(Y_Test,y_test_predicted))
scatterplot(X_Test,Y_Test, y_test_predicted, varname)

%Test with different alpha
for i =1:8
    fprintf("Alpha = %d\n" ,i)
    b = ridge(Y,X,i,0);
    y_pred_ridge = b(1) + X*b(2:end);
    R2_Ridge = rsquare(y_pred_ridge, Y);
    fprintf("R2_Ridge is: %f\n", R2_Ridge);
    [B,FitInfo] = lasso(X,Y,'lambda',i);
    y_pred_lasso = X*B+FitInfo.Intercept;
    R2_Lasso = rsquare(y_pred_lasso, Y);
    fprintf("R2_Lasso is: %f\n", R2_Lasso);
end


