function [Alive_test, X_test_NNvar] = KNN_Survival5(X_test,X_train,Alive_train,K,Beta1,Filters,sigma,Lambda)
%
% This predicts survival based on the labels of K-nearest neighbours 
% using weighted euclidian distance and the K-M estimator.
%
% INPUTS: 
% --------
% IMPORTANT: features as rows, samples as columns
% WARNING: Input data has to be normalized to have similar scales
% 
% X_test - testing sample features
% X_train - training sample features
% Alive_train - Alive dead status (+1 (alive) --> -1 (dead))
% K - number of nearest-neighbours to use
% Beta - shrinkage factor --> higher values indicate less important
%        features
% Filters - method of emphasizing or demphasizing neighbours 
% sigma - sigma of gaussian filter (lower values result in more emphasis on closes neighbours)
% Lambda -  the larger lambda, the less penalty on lack of common
%           dimensions (when there are no NAN values and lamda = 1, 
%           there's no dimension penlty)
%
% OUTPUTS:
% ---------
% Alive_test - Alive dead status (continuous: +1 (alive) --> -1 (dead))
% time - time indicator 
%
% X_test_NNvar - variance along each dimension of K-nearest neighbours of
%                each patient
%

%% Sample Inputs
% clear ; close all ; clc ; 
% 
% % seed random number generator for reproducibility
% %rng('default');
% 
% N_train = 100; %training sample size
% N_test = 10; %testing sample size
% p = 12; %no of features
% 
% %
% % FOR THE FOLLOWING: features in rows, samples on columns
% %
% X_train = randn(p,N_train); % training features (continuous)
% X_test = randn(p,N_test); % testing features (continuous)
% 
% % Add NAN values at random to simulate missing data
% pNaN = 0.2; %proportion of NAN values
% for i = 1:(pNaN * N_train*p)
% X_train(randi(p),randi(N_train)) = nan;
% end
% for i = 1:(pNaN * N_test*p)
% X_test(randi(p),randi(N_test)) = nan;
% end
% 
% t_min = 3;
% t_max = 302;
% 
% Survival_train = randi([t_min,t_max],1,N_train); % survival of training sample
% Censored_train = randi([0,1],1,N_train); % censorship of training sample: 1=alive
% %Convert outcome from survival to alive/dead status using time indicator
% Alive_train = TimeIndicator(Survival_train,Censored_train,t_min,t_max);
% 
% K = 15; % number of nearest-neighbours to use
% Beta1 = ones(p,1); %shrinkage factor for each feature
% 
% %Filters = 'Euclidian';
% %Filters = 'Gaussian';
% %Filters = 'Both';
% Filters = 'None';
% 
% sigma = 2*K; %sigma of gaussian filter (lower values result in more emphasis on closes neighbours)
% 
% Lambda = 1; % the larger lambda, the less penalty on lack of common dimensions
%             % when there are no NAN values and lamda = 1, there's no dimension penlty

%% Begin algorithm

% initialize
Alive_test = nan(length(Alive_train(:,1)),length(X_test(1,:)));
P_Center_Max = length(X_test(1,:));
X_test_NNvar= nan(size(X_test)); %Nearest neighbours' variance along each feature

for P_Center = 1:P_Center_Max

% current point to label
Center = X_test(:,P_Center);

%% Compare to every other point using non-missing common features 

% initializa distance to each surrounding point
Dist = zeros(2,length(X_train(1,:))); %distnace
Dist(2,:) = 1:length(X_train(1,:)); %sample index

P_SurroundMax = length(Dist);

for P_Surround = 1:P_SurroundMax
    Surround = X_train(:,P_Surround);
    
    % find common dimensions
    Keep_feat = ~isnan(Center) & ~isnan(Surround);
    Center_temp = Center(Keep_feat==1);
    Surround_temp = Surround(Keep_feat==1);
    Ndim = length(Center_temp) ./ length(Center); % proportion of total dimensions kept
    % don't shrink features if some un-shared dimensions
    Beta1_temp = Beta1;
    if length(Center) > length(Center_temp)
    Beta1_temp = ones(length(Center_temp),1);
    end
    % Weighted euclidian distance
    Dist(1,P_Surround) = (sum((Beta1_temp.^2) .* (abs(Center_temp - Surround_temp)))) ./ (Lambda * Ndim);
    
end

% Get nearest neighbours
Dist = sortrows(Dist',1);
Dist = Dist(1:K,:);

%% Get nearest neighbour variance

% get nearest neighbour features
NNs = X_train(:,Dist(:,2)');
% find number of NN's having each feature
nonNAN_count = sum(~isnan(NNs),2);
%find nonNAN sum along each feature
nonNAN_sum = NNs;
nonNAN_sum(isnan(nonNAN_sum)==1) = 0;
nonNAN_sum = sum(nonNAN_sum,2);
% find nonNAN mean along each feature
nonNAN_mean = nonNAN_sum ./ nonNAN_count;
% Get nonNAN variance along each feature
[~,nonNAN_mean_mesh] = meshgrid(1:length(NNs(1,:)),nonNAN_mean);
nonNAN_var = (NNs - nonNAN_mean_mesh) .^ 2;
nonNAN_var(isnan(nonNAN_var)==1) = 0;
nonNAN_var = sum(nonNAN_var,2) ./ nonNAN_count;
% penalizing comparisons among fewer neghbours. Since more variability is naturally expected if there were more neighbours.
% dividing by non_NAN count so if fewer neighbours were being compared, variance would increase to compensate
nonNAN_var = nonNAN_var ./ nonNAN_count;
X_test_NNvar(:,P_Center) = nonNAN_var;

%% Get alive/dead status of K nearest neighbours
Alive_surround = Alive_train(:,Dist(:,2));

%% Get emphasizing filters and apply to nearest neighbours

% Source note: the gaussian filter code was taken
% from http://stackoverflow.com/questions/6992213/gaussian-filter-on-a-vector-in-matlab

% Gaussian filter
if strcmp(Filters,'Gaussian') == 1 || ...
        strcmp(Filters,'Both') == 1
    Size = 2*K;
    x = linspace(-Size / 2, Size / 2, Size);
    gaussFilter = exp(-x .^ 2 / (2 * sigma ^ 2));
    gaussFilter = gaussFilter / sum (gaussFilter); % normalize
    gaussFilter = gaussFilter(1:K);
    gaussFilter(1:K) = 2.*(gaussFilter(1:K));
    gaussFilter = gaussFilter';
    gaussFilter = sort(gaussFilter,'descend');
end
% inverse euclidian filter
if strcmp(Filters,'Euclidian') == 1 || ...
        strcmp(Filters,'Both') == 1
    euclFilter = 1- (Dist(:,1) ./ sum(Dist(:,1)));
    euclFilter = euclFilter ./ sum(euclFilter);
    
end
% final filter
if strcmp(Filters,'Euclidian') == 1 
    Filter = euclFilter;
elseif strcmp(Filters,'Gaussian') == 1
    Filter = gaussFilter;
elseif strcmp(Filters,'Both') == 1
    Filter = euclFilter .* gaussFilter;
elseif strcmp(Filters,'None') == 1
    Filter = ones(length(Dist(:,1)),1);
end

% normalize and meshgrid
Filter = Filter' ./ sum(Filter);
[Filter,~] = meshgrid(Filter,1:length(Alive_train(:,1)));

% apply filter to nearest neighbours
Alive_surround = Alive_surround .* Filter;


%% Get alive/dead status for central point (continuous)

UnknownStatus = isnan(Alive_surround);
Alive_surround(isnan(Alive_surround)==1) = 0;
% average alive/dead status at each time point (ignoring unknown status)
Alive_center = sum(Alive_surround, 2) ./ (K - sum(UnknownStatus, 2));

% save result
Alive_test(:,P_Center) = Alive_center;

end

% normalize
Surv_max = max(Alive_test);
[Surv_max,~] = meshgrid(Surv_max,1:length(Alive_test(:,1)));
Alive_test = Alive_test ./ Surv_max;

end