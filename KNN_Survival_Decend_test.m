function Beta_star = KNN_Survival_Decend_test(X_valid,Survival_valid,Censored_valid,X_train,Survival_train,Censored_train,K,Gamma,Pert)
%
% This predicts survival based on the labels of K-nearest neighbours 
% using weighted euclidian distance and the K-M estimator.
%
% INPUTS: 
% --------
% IMPORTANT: features as rows, samples as columns
% WARNING: Input data has to be normalized to have similar scales
% 
% X_valid - testing sample features
% Survival_valid - survival of validation sample
% Censored_valid - censorship of validation sample: 1 = ALIVE, 0 = DEAD
% X_train - training sample features
% Survival_train - survival of training sample
% Censored_train - censorship of training sample: 1 = ALIVE, 0 = DEAD
% K - number of nearest-neighbours to use
% Gamma = 0.1; % learning rate
% Pert = 0.01; % this controls how much to perturb beta each time to calculate gradient

%
% OUTPUTS:
% ---------
% Beta_star - shrinkage factor that gives least error
%

%% Calculate true PAUC using KM estimator

[t,f,~,~] = KM(Survival_valid',Censored_valid');
if sum(Censored_valid) < length(Censored_valid)-1
pAUC_true = sum(diff(t) .* f(1:end-1,:)) / sum(diff(t)); %proportion of area under curve covered
elseif sum(Censored_valid) >= length(Censored_valid)-1 %almost all surrounding points are censored
pAUC_true = 1;
end

%%

[p,~] = size(X_train);

% Initialize beta
%Beta0 = randn(p,1); %random initialization
Beta0 = ones(p,1); %all features carry equal weight

% Apply KNN using current beta
Y_valid0 = KNN_Survival2(X_valid,X_train,Survival_train,Censored_train,K,Beta0);
% Calculate pAUC under KM estimator
[t,f,~,~] = KM(Y_valid0',zeros(size(Censored_valid')));
pAUC_estimate = sum(diff(t) .* f(1:end-1,:)) / sum(diff(t)); %proportion of area under curve covered
% Calculate cost (error)
Cost0 = (pAUC_true - pAUC_estimate)^2;

Beta_star = Beta0;
Cost_star = Cost0;


for progress = 1:20
    
    % Apply KNN using current beta
    Y_valid0 = KNN_Survival2(X_valid,X_train,Survival_train,Censored_train,K,Beta0);
    % Calculate pAUC under KM estimator
    [t,f,~,~] = KM(Y_valid0',zeros(size(Censored_valid')));
    pAUC_estimate = sum(diff(t) .* f(1:end-1,:)) / sum(diff(t)); %proportion of area under curve covered
    % Calculate cost (error)
    Cost0 = (pAUC_true - pAUC_estimate)^2;
    
    Cost_All(progress,1) = Cost0; %record cost
    
    Gradient = zeros(size(Beta0));
    for j = 1:5
    % Perturb beta at little to get a feeling for gradient
    Beta1 = Beta0;
    Beta1(1,1) = Beta1(1,1) + Pert.*randn(1);
    %Beta1 = Beta0 + Pert.*randn(p,1);
    %Beta1 = Beta0 + Pert;
    % Apply KNN using perturbed beta
    Y_valid1 = KNN_Survival2(X_valid,X_train,Survival_train,Censored_train,K,Beta1);
    % Calculate pAUC under KM estimator
    [t,f,~,~] = KM(Y_valid1',zeros(size(Censored_valid')));
    pAUC_estimate = sum(diff(t) .* f (1:end-1,:)) / sum(diff(t)); %proportion of area under curve covered
    % Calculate cost (error)
    Cost1 = (pAUC_true - pAUC_estimate)^2;
    
    % Calculate gradient
    deltaC = Cost1 - Cost0;
    deltaBeta = Beta1 - Beta0;
    Gradient_j = deltaC ./ deltaBeta;
    Gradient = Gradient + Gradient_j;

    Gradient = Gradient ./ j;
    end
    
    % Gradient descent
    Beta0 = Beta0 - (Gamma .* Gradient);
    
    % NEW!!!
    Beta0(2:end) = 1; %to get rid of infinities since only first value of beta was changed
    
    % Apply KNN using updated beta
    Y_valid_New = KNN_Survival2(X_valid,X_train,Survival_train,Censored_train,K,Beta0);
    % Calculate pAUC under KM estimator
    [t,f,~,~] = KM(Y_valid_New',zeros(size(Censored_valid')));
    pAUC_estimate = sum(diff(t) .* f(1:end-1,:)) / sum(diff(t)); %proportion of area under curve covered
    % Calculate cost (error)
    Cost_New = (pAUC_true - pAUC_estimate)^2;
    
    if Cost_New < Cost_star
        Beta_star = Beta0;
        Cost_star = Cost_New;
    end
    
end

plot(Cost_All)

end