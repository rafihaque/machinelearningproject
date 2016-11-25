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
Survival = Survival(Keep1) +3; %add 3 to ignore negative survival
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

%% Assess performance of basic model using  train/validation approach %%%%%

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

t_min = min(Survival)-1;
t_max = max(Survival);
time = [t_min:1:t_max]';
% Convert outcome from survival to alive/dead status using time indicator
Alive_train = TimeIndicator(Survival_train,Censored_train,t_min,t_max);
Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);
Alive_test = TimeIndicator(Survival_test,Censored_test,t_min,t_max);

%% K-NN survival parameters
K = 30;
Beta = ones(length(X_train(:,1)),1); %shrinkage factor for each feature

%Filters = 'Euclidian';
%Filters = 'Gaussian';
Filters = 'Both';
%Filters = 'None';

Gamma = 10; %learning rate
Pert = 0.1; %this controls how much to perturb beta to get a feeling for gradient
Conv_Thresh = 0.0005; %convergence threshold for sigma
sigma_init = 3; %ceil(K/3); %initial sigma

%% Get optimum sigma value using gradient descent

sigma = KNN_Survival_Decend2a(X_valid,X_train,Alive_train,Alive_valid,K,Beta,Filters,Gamma,Pert,Conv_Thresh,sigma_init);


%% Apply optimal sigma and calculate c-index

Alive_test_hat = KNN_Survival3(X_test,X_train,Alive_train,K,Beta,Filters,sigma);
Alive_test_hat = sum(Alive_test_hat);
Optimal_C = cIndex2(Alive_test_hat,Survival_test,Censored_test)

%% Compare to non-optimal sigma's error

Alive_test_hat = KNN_Survival3(X_test,X_train,Alive_train,K,Beta,Filters,sigma_init);
Alive_test_hat = sum(Alive_test_hat);
original_C = cIndex2(Alive_test_hat,Survival_test,Censored_test)