%
% Using Cox proportional hazards regression with elastic net to predict
% survival
%

clear ;
clf;
%close all ;
clc ; 

addpath('./Codes');
addpath('./Codes/glmnet_matlab/glmnet_matlab');

%% Load data

load('ClinicalData.mat');

Data = ClinicalData.Data;
Survival = ClinicalData.Survival;
Censored = ClinicalData.Censored;
FeatureNames = ClinicalData.FeatureNames;
Patients = ClinicalData.Patients;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data = Data(1:20,:);
% Survival = Survival(1:20,:);
% Censored = Censored(1:20,:);
% Patients = Patients(1:20,:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%------
% Data = ProteomicData.Data;
% Survival = ProteomicData.Survival;
% Censored = ProteomicData.Censored;
% FeatureNames = ProteomicData.FeatureNames;
% Patients = ProteomicData.Patients;
%------
% Data = GenomicData.Data;
% Survival = GenomicData.Survival;
% Censored = GenomicData.Censored;
% FeatureNames = GenomicData.FeatureNames;
% Patients = GenomicData.Patients;

%% Putting testing sample aside
kMax = 100;
Perc_test = 0.4; %proportion allocated to testing sample

c_train = zeros(kMax,1);
c_test = zeros(kMax,1);

for k = 1:kMax

    N_all = length(Data(:,1));
    N_test = ceil(Perc_test*N_all); %testing sample size
    % indices of testing samples
    Idx_temp = (randperm(N_all))'; %random assignment
    Idx_test = Idx_temp(1:N_test);
    Idx_train = Idx_temp(N_test+1:end);

    Data1 = Data;
    Survival1 = Survival;
    Censored1 = Censored;
    Patients1 = Patients;

    Data_test = Data(Idx_test,:);
    Survival_test = Survival(Idx_test,:);
    Censored_test = Censored(Idx_test,:);
    Patients_test = Patients(Idx_test,:);

    Data_train = Data(Idx_train,:);
    Survival_train = Survival(Idx_train,:);
    Censored_train = Censored(Idx_train,:);
    Patients_train = Patients(Idx_train,:);

    % j = 0; 
    % for i = 1:length(Idx_test(:,1))
    %     Idx_current = Idx_test(i,1);     
    % 
    %     Data_test(i,:) = Data(i-j,:); 
    %     Survival_test(i,:) = Survival(i-j,:); 
    %     Censored_test(i,:) = Censored(i-j,:); 
    %     Patients_test(i,:) = Patients(i-j,:);
    %
    %     Data(i-j,:) = []; 
    %     Survival(i-j,:) = [];
    %     Censored(i-j,:) = [];
    %     Patients(i-j,:) = [];
    % 
    %     j = j+1; 
    % end

    % saving test sample structure
    TestingData.Data = Data_test;
    TestingData.Survival = Survival_test;
    TestingData.Censored = Censored_test;
    TestingData.Patients = Patients_test;
    %% Cox prop. hazards regression with Elastic Net
    % reverse notation for censorship status (NOTE THAT THE GLMNET FUNCTION
    % USES 1 FOR DEATH AND 0 FOR CENSORSHIP)
    Censored_train = Censored_train+1;
    Censored_train(Censored_train==2) = 0;
    Censored_test = Censored_test+1;
    Censored_test(Censored_test==2) = 0;
    
    X = Data_train;
    Y = [Survival_train,Censored_train];

    % % Without cross-validation
    % fit=glmnet(X,Y,'cox');
    % glmnetPlot(fit);
    % With cross-validation
    K = 10; %how many folds?
    cvfit=cvglmnet(X,Y,'cox',[],[],K);
    %cvglmnetPlot(cvfit);
    % Make predictions using model
    % pred1 = glmnetPredict(fit,X); %without CV
    % pred1 = cvglmnetPredict(cvfit,X); %with CV
    %% Getting optimal model coefficients
    % Extract index of optimum lambda
    lambda_optimum = cvfit.lambda_1se; %maximum lambda within one SE of that which minimizes CV error
    %lambda_optimum = cvfit.lambda_min;
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

    %% Calculating Concordance index for chosen model (evaluated on testing sample)
    % training concordance (reverse training error)
    c_train(k,1) = cIndex(Beta, Data_train, Survival_train, Censored_train);
    % testing concordance (reverse testing error)
    c_test(k,1) = cIndex(Beta, Data_test, Survival_test, Censored_test);
end

boxplot([c_train,c_test],'Labels',{'cIndex_train','cIndex_test'}); 
hold on ; 
title(['Training Vs Testing cIndex at ' num2str(1-Perc_test) ' / ' num2str(Perc_test) ' train-test split']); 
ylabel('cIndex');

