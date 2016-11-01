%
% This fetches the most relevant protein expression (proteomic) features
% from LGG data
%

clear ; close all ; clc ; 

%% Load LGG data

load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/GBMLGG.Data.mat')
Symbols = cellstr(Symbols);
Samples = cellstr(Samples);

%% Extract proteomic features
Idx_Start = 230; % start of proteomic features
Idx_End = 446; %end of proteomic features

Proteomic = Features(Idx_Start:Idx_End,:)';
Symbols = Symbols(Idx_Start:Idx_End,:)';

Survival = Survival';
Censored = Censored';

Proteomic = [Proteomic,Survival,Censored];

%% Remove patients with lots of missing features

MissingThr_Pat = 0.2; %threshold of missing values after which patients are discarded

Proteomic1 = Proteomic;
j = 0;
for i = 1:length(Proteomic(:,1))
   
    dummy = Proteomic1(i,:);
    dummy(isnan(dummy)~=1) = 0;
    dummy(isnan(dummy)==1) = 1;
    
    % remove patients with missing survival or censorship status or more
    % than the threshold number of features
    if sum(dummy(1,:)) > (MissingThr_Pat * length(Proteomic(1,:))) || ...
            isnan(Proteomic1(i,end-1))==1 || isnan(Proteomic1(i,end))==1
        
        Proteomic(i-j,:) = [];
        Samples(i-j,:) = [];
        j = j+1; %since when you delete a feature the index of the "Proteomics" matrix shifts by one
    end
    
    
end


%% Remove features with lots of missing data or who are right-censored

MissingThr_Feat = 0.2; %threshold of missing values after which features are discarded

Proteomic1 = Proteomic;
j = 0;
for i = 1:length(Proteomic(1,:))-2 %(-2 to ignore survival and censored status from feature removal)
    
    dummy = Proteomic1(:,i);
    dummy(isnan(dummy)~=1) = 0;
    dummy(isnan(dummy)==1) = 1;
    
    if sum(dummy(:,1)) > (MissingThr_Feat * length(Proteomic(:,1)))
        Proteomic(:,i-j) = [];
        Symbols(:,i-j) = [];
        j = j+1; %since when you delete a feature the index of the "Proteomics" matrix shifts by one
    end
    
end

Survival = Proteomic(:,end-1);
Censored = Proteomic(:,end);
Proteomic = Proteomic(:,1:end-2);

%% A bit more preprocessing

% Z-score standardization so that features are comparable
for i = 1:length(Proteomic(1,:))
    Mu = mean(Proteomic(:,i));
    StDev = std(Proteomic(:,i));
    
    Proteomic(:,i) = (Proteomic(:,i)-Mu)./StDev;
end

% Delete features with NAN values becuase their StDev was 0
Proteomic1 = Proteomic;
j = 0;
for i = 1:length(Proteomic1(1,:))
    
    if isnan(max(Proteomic1(:,i))) == 1 
        Proteomic(:,i-j) = [];
        Symbols(:,i-j) = [];
        j = j+1; %since when you delete a feature the index of the "mutations" matrix shifts by one
    end
end

% Add 2 to survival to ignore negative values
Survival = Survival+2;


%% Write output structure

ProteomicData.Patients = Samples;
ProteomicData.FeatureNames = Symbols;
ProteomicData.Data = Proteomic;
ProteomicData.Survival = Survival;
ProteomicData.Censored = Censored;