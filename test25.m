%
% KNN MODEL INTERPRETATION !!!
%

% add relevant paths
clear; close all; clc;
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Results/Feature_reduction/GBMLGG/')

% turn off warnings
% warning('off','all')

%% Choose which model to use

%WhichModel = 'Basic';
WhichModel = 'Reduced';
%WhichModel = 'Unprocessed';

if strcmp(WhichModel, 'Basic') == 1
load 'BasicModel.mat';
Features = BasicModel.Features;
Survival = BasicModel.Survival +3; %add 3 to ignore negative survival
Censored = BasicModel.Censored;

elseif strcmp(WhichModel, 'Reduced') == 1
load 'ReducedModel.mat';
Features = ReducedModel.Features;
Survival = ReducedModel.Survival +3; %add 3 to ignore negative survival
Censored = ReducedModel.Censored;

elseif strcmp(WhichModel, 'Unprocessed') == 1
load 'GBMLGG.Data.mat';
Survival = Survival +3; %add 3 to ignore negative survival
end

% remove NAN survival or censorship values
Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];

Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

% NEW!!! REMOVE PROTEIN AND mRNA FEATURES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Features(182:end,:) = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[p,N] = size(Features);

%% Determine initial parameters

%K_min = 15; 
%K_max = 70;
K = 30;

Filters = 'None';
%Filters = 'Both'; %choose this if performing gradient descent on sigma

Beta_init = ones(length(Features(:,1)),1); %initial beta (shrinking factor for features)
sigma_init = 7;

Lambda = 1; %the less the higher penality on lack of common dimensions

% Parameters for gradient descent on beta
Gamma_Beta = 15; %learning rate
Pert_Beta = 5; %this controls how much to perturb beta to get a feeling for gradient
Conv_Thresh_Beta = 0.0001; %convergence threshold 

Gamma_sigma = 10; %learning rate
Pert_sigma = 0.1; %this controls how much to sigma beta to get a feeling for gradient
Conv_Thresh_sigma = 0.0005; %convergence threshold for sigma

%Descent = 'None'; %fast
Descent = 'Beta'; %slow, especially with more features
%Descent = 'sigma'; %slow, especially with more features

trial_No = 50; % no of times to shuffle

%%

C = zeros(trial_No,1);

trial = 1;
%for trial = 1 %:trial_No

    %% Shuffle samples

    Idx_New = randperm(N,N);
    Features_New = zeros(size(Features));
    Survival_New = zeros(size(Survival));
    Censored_New = zeros(size(Censored));
    for i = 1:N
    Features_New(:,i) = Features(:,Idx_New(1,i));
    Survival_New(:,i) = Survival(:,Idx_New(1,i));
    Censored_New(:,i) = Censored(:,Idx_New(1,i));
    end
    Features = Features_New;
    Survival = Survival_New;
    Censored = Censored_New;

    %% Assign samples to PROTOTYPE set, validation set 
    %  The reason we call it "prototype set" rather than training set is 
    %  because there is no training involved. Simply, the patients in the 
    %  validation/testing set are matched to similar ones in the prototype
    %  ("database") set.
    
    K_cv = 2;
    Folds = ceil([1:N] / (N/K_cv));

    X_prototype = Features(:, Folds == 1);
    X_valid = Features(:, Folds == 2);

    Survival_prototype = Survival(:, Folds == 1);
    Survival_valid = Survival(:, Folds == 2);

    Censored_prototype = Censored(:, Folds == 1);
    Censored_valid = Censored(:, Folds == 2);

    % Convert outcome from survival to alive/dead status using time indicator
    t_min = min(Survival)-1;
    t_max = max(Survival);
    time = [t_min:1:t_max]';
    Alive_prototype = TimeIndicator(Survival_prototype,Censored_prototype,t_min,t_max);
    Alive_valid = TimeIndicator(Survival_valid,Censored_valid,t_min,t_max);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Begin algorithm

% initialize beta
Beta0 = Beta_init; %shrinkage factor for each feature
    
%% Get cost for initialized beta
% predicted survival
Alive_valid_hat0 = KNN_Survival3(X_valid,X_prototype,Alive_prototype,K,Beta0,Filters,sigma_init);
% cost over each time
Cost0 = (Alive_valid - Alive_valid_hat0).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost0);
Cost0(Cost_nan == 1) = 0;
Cost0 = sum(Cost0);
Cost0 = Cost0 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% Start gradient descent till convergence

Beta_star = Beta0;
Cost_star = Cost0;

step = 0;
Convergence = 0;
while Convergence == 0 

step = step + 1;    

% Uncomment the following to monitor progress
clc
step
%BETA0 = Beta0'
%BETA_STAR = Beta_star'
COST0 = mean(Cost0)
COST_STAR = mean(Cost_star)


%% Find gradient with respect to each component in beta

Gradient = zeros(size(Beta0));

for i = 1:length(Beta0)


%% Perturb beta component and calculate new cost

Beta1 = Beta0;
Beta1(i,1) = Beta0(i,1) + Pert_Beta;

% predicted survival
Alive_valid_hat1 = KNN_Survival3(X_valid,X_prototype,Alive_prototype,K,Beta1,Filters,sigma_init);
% cost over each time
Cost1 = (Alive_valid - Alive_valid_hat1).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost1);
Cost1(Cost_nan == 1) = 0;
Cost1 = sum(Cost1);
Cost1 = Cost1 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% Calculate gradient

deltaCost = Cost1 - Cost0;
deltaBeta = Beta1(i,1) - Beta0(i,1);
gradient = deltaCost ./ deltaBeta;

% Calculate average gradient
Gradient(i,1) = mean(gradient);

end


%% Update beta
Beta0 = Beta0 - (Gamma .* Gradient);

%% Get cost for new beta

% predicted survival
Alive_valid_hat1 = KNN_Survival3(X_valid,X_prototype,Alive_prototype,K,Beta0,Filters,sigma_init);
% cost over each time
Cost1 = (Alive_valid - Alive_valid_hat1).^2;
% average cost for each sample (excluding nan's in comparison)
Cost_nan = isnan(Cost1);
Cost1(Cost_nan == 1) = 0;
Cost1 = sum(Cost1);
Cost1 = Cost1 ./ (length(Cost_nan(:,1))-sum(Cost_nan));

%% Get difference in cost and act accordingle

% update optimum beta
if mean(Cost1) < mean(Cost_star)
    Beta_star = Beta0;
    Cost_star = Cost1;
end

% determine if convergence reached
if abs( mean(Cost1) - mean(Cost0) ) > Conv_Thresh
    Cost0 = Cost1;
elseif abs( mean(Cost1) - mean(Cost0) ) <= Conv_Thresh
    Convergence = 1; 
end

% exit if takes too long to converge
if step > 40
    Convergence = 1;
end

end