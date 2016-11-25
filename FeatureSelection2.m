clear; close all; %clc;

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

%% Assess performance of basic model using  K-fold cross validation %%%%%%%

%define number of cross-validation folds and assign samples
K_cv = 10;
Folds = ceil([1:N] / (N/K_cv));

% K-NN survival parameters
K = 30;
Beta1 = ones(length(Basic(:,1)),1); %all features carry equal weight
%Beta1 = [1;0.7;0.4;1.5;1]; %tunable weight parameters

fprintf('Basic Model assessment with %d-fold cross-validation.\n', K_cv);

%cycle through folds, training and testing model
C = nan(1,K_cv);

for i = 1:K_cv

    X_valid = Basic(:, Folds == i);
    X_train = Basic(:, Folds ~= i);

    Survival_train = Survival(:, Folds ~= i);
    Censored_train = Censored(:, Folds ~= i);

    Y_valid = KNN_Survival2(X_valid,X_train,Survival_train,Censored_train,K,Beta1);
    %Y_train = KNN_Survival(X_train,X_train,Survival_train,Censored_train,K);

    Survival_valid = Survival(:, Folds == i);
    Censored_valid = Censored(:, Folds == i);

    %Classification-based survival%%%%%%%%%%%%%%%%%
%     q33 = quantile(Y_valid,0.33);
%     q66 = quantile(Y_valid,0.66);
%     Y_valid(Y_valid<q33) = 1;
%     Y_valid(Y_valid>=q33 & Y_valid<q66) = 2;
%     Y_valid(Y_valid>=q66) = 3;
%     
%     q33 = quantile(Survival_valid,0.33);
%     q66 = quantile(Survival_valid,0.66);
%     Survival_valid(Survival_valid<q33) = 1;
%     Survival_valid(Survival_valid>=q33 & Survival_valid<q66) = 2;
%     Survival_valid(Survival_valid>=q66) = 3;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    C(i) = cIndex2(Y_valid,Survival_valid,Censored_valid);
    %C(i) = cIndex2(Y_train,Survival_train,Censored_train);
end

fprintf('\tmean c-index = %g, standard deviation = %g\n', mean(C), std(C));