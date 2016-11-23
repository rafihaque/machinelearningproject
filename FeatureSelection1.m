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
% K = 10;
% Basic = KNN_Impute(Basic,K,'Regression');

%convert symbols, symbol types to cell array
Symbols = cellstr(Symbols);
SymbolTypes = cellstr(SymbolTypes);

%define core set of samples that have all basic features, survival, censoring
Keep1 = ~isnan(Survival) & ~isnan(Censored) ...
    & (sum(isnan(Basic), 1) == 0);
Basic = Basic(:, Keep1);
Features = Features(:, Keep1);
Survival = Survival(Keep1);
Censored = Censored(Keep1);
N = length(Survival);

%% Assess performance of basic model using  10-fold cross validation %%%%%%%
%%
%define number of folds and assign samples
K = 10;
Folds = ceil([1:N] / (N/K));

fprintf('Basic Model assessment with %d-fold cross-validation.\n', K);

%cycle through folds, training and testing model
C = nan(1,K);
for i = 1:K
    Beta = coxphfit(Basic(:, Folds ~= i).', Survival(Folds ~= i).',...
        'Censoring', Censored(Folds ~= i).');
    C(i) = cIndex(Beta, Basic(:, Folds == i).', Survival(Folds == i),...
        Censored(Folds == i));
end
fprintf('\tmean c-index = %g, standard deviation = %g\n', mean(C), std(C));


%% Adding any other highly relevant CLINICAL features to the model (if any).
%  Method used: forward stepwise feature selection with (??????)
%%

%
% STEP 1: DEALING WITH MISSING DATA - THREE ALTERNATIVES
%

% ALTERNATIVE 1:
% %Deleting all samples that miss any of the clinical features
% Keep2 = sum(isnan(Features(strcmp(SymbolTypes, 'Clinical'),:)), 1) == 0;
% Model2 = Basic(:, Keep2);
% Features2 = Features(:, Keep2);
% Survival2 = Survival(Keep2);
% Censored2 = Censored(Keep2);
% N = length(Survival2);

% ALTERNATIVE 2:
% % Imputing clinical values by adding "default" values in empty locations
% % (Modes for binary variables)
% ClinicalFeatures = Features(strcmp(SymbolTypes, 'Clinical'),:);
% Modes2 = mode(ClinicalFeatures,2);
% for i = 1:length(Modes2)
% CurrentFeature = ClinicalFeatures(i,:);
% CurrentFeature(isnan(CurrentFeature)==1) = Modes2(i,1);
% ClinicalFeatures(i,:) = CurrentFeature;
% end
