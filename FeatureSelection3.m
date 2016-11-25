clear; close all; clc;

% NOTE: Codes used here are a combination of original codes and codes
% provided in the "PerformanceExample.m" script provided by Dr Lee Cooper
% in CS-534 class
% add relevant paths
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')

% turn off warnings
% warning('off','all')

%% Using basic model as provided by Dr Lee (PerformanceExample.m) for a starter
%%
% read in data and extract minimal features
load 'GBMLGG.Data.mat';

%define a basic model containing IDH mutations, age and chromosomes 1p/19q
x1 = Features(strcmp(cellstr(Symbols),...
    'age_at_initial_pathologic_diagnosis_Clinical'), :);
x2 = Features(strcmp(cellstr(Symbols), 'IDH1_Mut'), :);
x3 = Features(strcmp(cellstr(Symbols), 'IDH2_Mut'), :);
x4 = Features(strcmp(cellstr(Symbols), '1p_CNVArm'), :);
x5 = Features(strcmp(cellstr(Symbols), '19q_CNVArm'), :);

% NOTE: Z-score standardization does not affect c-index
% Z-score standardization of continuous features
x1 = (x1 - mean(x1(~isnan(x1)))) ./ std(x1(~isnan(x1))); %age
x4 = (x4 - mean(x4(~isnan(x4)))) ./ std(x4(~isnan(x4))); %1p_CNVArm
x5 = (x5 - mean(x5(~isnan(x5)))) ./ std(x5(~isnan(x5))); %19q_CNVArm
Basic = [x1; x2; x3; x4; x5];

% NOTE: imputation of basic model gives lower c-index (75% Vs 78%)!
% Imputation of missing values in basic model using KNN
%K = 10;
%Basic = KNN_Impute(Basic,K,'Regression');

%convert symbols, symbol types to cell array
Symbols = cellstr(Symbols);
SymbolTypes = cellstr(SymbolTypes);

%define core set of samples that have all basic features, survival, censoring
Keep1 = ~isnan(Survival) & ~isnan(Censored) ...
    & (sum(isnan(Basic), 1) == 0);
Basic = Basic(:, Keep1);
Features = Features(:, Keep1);
Survival = Survival(Keep1) +2; %add 2 to ignore negative survival
Censored = Censored(Keep1);
N = length(Survival);

%% Shuffle samples

Idx_New = randperm(N,N);
Basic_New = zeros(size(Basic));
Survival_New = zeros(size(Survival));
Censored_New = zeros(size(Censored));
for i = 1:N
Basic_New(:,i) = Basic(:,Idx_New(1,i));
Survival_New(:,i) = Survival(:,Idx_New(1,i));
Censored_New(:,i) = Censored(:,Idx_New(1,i));
end
Basic = Basic_New;
Survival = Survival_New;
Censored = Censored_New;

%% Assess performance of basic model using  K-fold cross validation %%%%%%%

% Assign samples to training, testing and validation
K_cv = 3;
Folds = ceil([1:N] / (N/K_cv));
 
X_train = Basic(:, Folds == 1);
X_valid = Basic(:, Folds == 2);
X_test = Basic(:, Folds == 3);

Survival_train = Survival(:, Folds == 1);
Survival_valid = Survival(:, Folds == 2);
Survival_test = Survival(:, Folds == 3);

Censored_train = Censored(:, Folds == 1);
Censored_valid = Censored(:, Folds == 2);
Censored_test = Censored(:, Folds == 3);
    
% K-NN survival parameters
K = 30;    

% Determine Beta using gradient descent
Gamma = 0.1; 
Pert = 0.01;
% perturb all betas at once
Beta_star = KNN_Survival_Decend(X_valid,Survival_valid,Censored_valid,X_train,Survival_train,Censored_train,K,Gamma,Pert);
% perturb only one beta (age)
%Beta_star = KNN_Survival_Decend_test(X_valid,Survival_valid,Censored_valid,X_train,Survival_train,Censored_train,K,Gamma,Pert);

%%

% Apply non-optimal model on testing set and calculate c-index
Y_test = KNN_Survival2(X_test,X_train,Survival_train,Censored_train,K,ones(size(Beta_star)));
C_original = cIndex2(Y_test,Survival_test,Censored_test)

% Apply optimized model on testing set and calculate c-index
Y_test = KNN_Survival2(X_test,X_train,Survival_train,Censored_train,K,Beta_star);
C_opt = cIndex2(Y_test,Survival_test,Censored_test)
