clear; close all; clc;

addpath('C:/Users/heyuan0755/Documents/GitHub/machinelearningproject/Codes/glmnet_matlab/glmnet_matlab')%glmet
addpath('C:/Users/heyuan0755/Documents/GitHub/machinelearningproject/Result');%data
addpath('C:/Users/heyuan0755/Documents/GitHub/machinelearningproject');%cIndex2

% turn off warnings
% warning('off','all')
 
%% Choose which model to use
 
%WhichModel = 'Basic';
WhichModel = 'Reduced';
 
if strcmp(WhichModel, 'Basic') == 1
load 'BRCA_ReducedData.mat';
Features = BRCA_ReducedData.Features;
addon = min(BRCA_ReducedData.Survival)+1;
Survival = BRCA_ReducedData.Survival +addon; %add 3 to ignore negative survival
Censored = BRCA_ReducedData.Censored;
 
elseif strcmp(WhichModel, 'Reduced') == 1
load 'BRCA_ReducedData.mat';
Features =  BasicModel_BRCA.Features;
addon = min(BasicModel_BRCA.Survival)+1;
Survival = BasicModel_BRCA.Survival +addon; %add 3 to ignore negative survival
Censored = BasicModel_BRCA.Censored;
end
%Clean up the nan value in survival & censored
Keep = ~isnan(Survival) & ~isnan(Censored) & (sum(isnan(Features), 1) == 0);
Features = Features(:, Keep);
Survival = Survival(Keep);
Censored = Censored(Keep);

%Flip censored & uncensored due to glmnet setup
Censored = Censored+1;
Censored(Censored==2) = 0;
[p,N] = size(Features);
K_min = 10;
K_max = 70;
 
 
trial_No = 10; % no of times to shuffle
 
%%
 
C = zeros(trial_No,1);
for trial = 1:trial_No
 
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
    K_cv = 2;
    Folds = ceil([1:N] / (N/K_cv));
 
    X_train = Features(:, Folds == 1);
    X_test = Features(:, Folds == 2);
 
    Survival_train = Survival(:, Folds == 1);
    Survival_test = Survival(:, Folds == 2);
 
    Censored_train = Censored(:, Folds == 1);
    Censored_test = Censored(:, Folds == 2);

       
        clc
        trial
        X = X_train;
        Y = [Survival_train;Censored_train];
        % With cross-validation
        k_fold = 10; %how many folds?
        cvfit=cvglmnet(X',Y,'cox',[],[],k_fold);
        
        %%
        
        % Extract index of optimal beta coefficients
%         lambda_optimum = cvfit.lambda_min;
%         Idx = (1:length(cvfit.lambda))';
%         Beta_Idx = cvfit.lambda - lambda_optimum;
%         Beta_Idx(Beta_Idx==0)=nan;
%         Beta_Idx(isnan(Beta_Idx)==0)=0;
%         Beta_Idx(isnan(Beta_Idx)==1)=1;
%         Beta_Idx = Beta_Idx .* Idx;
%         Beta_Idx = sum(Beta_Idx);
        
        %%
       
    Y_hat_test = cvglmnetPredict(cvfit, X_test', []);
    C(trial,1) = cIndex2(Y_hat_test, Survival_test, Censored_test);
    
end