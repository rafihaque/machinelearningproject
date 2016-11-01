%
% Using Cox proportional hazards regression with elastic net to predict
% survival
%

clear ; close all ; clc ; 

%% Load data
%load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/ClinicalData.mat')
%load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/ProteomicData.mat')
load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/GenomicData.mat')

% Data = ClinicalData.Data;
% Survival = ClinicalData.Survival;
% Censored = ClinicalData.Censored;
% FeatureNames = ClinicalData.FeatureNames;
% Data = ProteomicData.Data;
% Survival = ProteomicData.Survival;
% Censored = ProteomicData.Censored;
% FeatureNames = ProteomicData.FeatureNames;
Data = GenomicData.Data;
Survival = GenomicData.Survival;
Censored = GenomicData.Censored;
FeatureNames = GenomicData.FeatureNames;


%% Cox prop. hazards regression with Elastic Net

% reverse notation for censorship status (NOTE THAT THE GLMNET FUNCTION
% USES 1 FOR DEATH AND 0 FOR CENSORSHIP)
Censored = Censored+1;
Censored(Censored==2) = 0;

X = Data;
Y = [Survival,Censored];

% % Without cross-validation
% fit=glmnet(X,Y,'cox');
% glmnetPlot(fit);

% With cross-validation
K = 10; %how many folds?
cvfit=cvglmnet(X,Y,'cox',[],[],K);
cvglmnetPlot(cvfit);

% Make predictions using model
% pred1 = glmnetPredict(fit,X); %without CV
% pred1 = cvglmnetPredict(cvfit,X); %with CV

%% Getting optimal model coefficients

% Extract index of optimum lambda
%lambda_optimum = cvfit.lambda_1se; %maximum lambda within one SE of that which minimizes CV error
lambda_optimum = cvfit.lambda_min;

% Extract index of optimal beta coefficients
Idx = (1:length(cvfit.lambda))';
Beta_Idx = cvfit.lambda - lambda_optimum;
Beta_Idx(Beta_Idx==0)=nan;
Beta_Idx(isnan(Beta_Idx)==0)=0;
Beta_Idx(isnan(Beta_Idx)==1)=1;
Beta_Idx = Beta_Idx .* Idx;
Beta_Idx = sum(Beta_Idx);

% extract optimal beta coefficients
Beta = cvfit.glmnet_fit.beta(:,Beta_Idx);

%% Calculating Concordance index for chosen model

% reverse notation for censorship status (NOTE THAT THE cIndex FUNCTION
% USES 0 FOR DEATH AND 1 FOR CENSORSHIP)
Censored = Censored+1;
Censored(Censored==2) = 0;

c = cIndex(Beta, X, Survival, Censored)