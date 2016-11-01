%
% This fetches the most relevant clinical features
% from GBMLGG data (and adds some)
%

clear ; close all ; clc ; 

%% Load LGG data

load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/GBMLGG.Data.mat')
Symbols = cellstr(Symbols);
Samples = cellstr(Samples);

%% Extract genomic features

Idx_Start = 1; %start of clinical features
Idx_End = 13; %end of clinical features
 
Clinical = Features(Idx_Start:Idx_End,:)';
Symbols = Symbols(Idx_Start:Idx_End,:)';

Survival = Survival';
Censored = Censored';

%% Load Subtype data

Subs = tdfread('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/glioma.subtypes.txt');

% initializing molecular subtype features
Symbols{1,Idx_End+1} = 'molecular_subtype-is-IDHwt_Clinical';
Symbols{1,Idx_End+2} = 'molecular_subtype-is-IDHmutnonCodel_Clinical';
Symbols{1,Idx_End+3} = 'molecular_subtype-is-IDHmutCodel_Clinical';

Clinical(:,Idx_End+1:Idx_End+3) = nan(length(Clinical(:,1)),3);

ClinicalData = [Samples,num2cell(Clinical),num2cell(Survival),num2cell(Censored)];
SubtypeData = [cellstr(Subs.Case),cellstr(Subs.IDH_codel_subtype)];

% Unifying name syntax
for i = 1:length(ClinicalData(:,1))
    ClinicalData{i,1} = ClinicalData{i,1}(1:12);
end

% sorting clinical data by patient name
ClinicalData = sortrows(ClinicalData,1);
% sorting subtype data by patient name
SubtypeData = sortrows(SubtypeData,1);

% Separating subtypes
IDHwt = SubtypeData;
nonCodel = SubtypeData;
Codel = SubtypeData;

for i = 1:length(SubtypeData(:,1))
    % IDH wt
    if strcmp(IDHwt{i,2},'IDHwt') == 1
        IDHwt{i,2} =1;
    elseif strcmp(IDHwt{i,2},'IDHmut-non-codel') == 1 || strcmp(IDHwt{i,2},'IDHmut-codel') == 1
        IDHwt{i,2} =0;
    else
        IDHwt{i,2} = nan;
    end
    % IDHmut-nonCodel
    if strcmp(nonCodel{i,2},'IDHmut-non-codel') == 1
        nonCodel{i,2} =1;
    elseif strcmp(nonCodel{i,2},'IDHwt') == 1 || strcmp(nonCodel{i,2},'IDHmut-codel') == 1
        nonCodel{i,2} =0;
    else
        nonCodel{i,2} = nan;
    end   
    % IDHmut-Codel
    if strcmp(Codel{i,2},'IDHmut-codel') == 1
        Codel{i,2} =1;
    elseif strcmp(Codel{i,2},'IDHwt') == 1 || strcmp(Codel{i,2},'IDHmut-non-codel') == 1
        Codel{i,2} =0;
    else
        Codel{i,2} = nan;
    end   
end

%% Add subtype features to clinical data

% map 'subtype patients' to 'clinical patients'
Mapping = StringMatch(ClinicalData(:,1), IDHwt(:,1));

for i = 1:length(Mapping(1,:))
    
    ClinicalData{i,Idx_End+2} = IDHwt{Mapping{1,i}(1,1),2};
    ClinicalData{i,Idx_End+3} = nonCodel{Mapping{1,i}(1,1),2};
    ClinicalData{i,Idx_End+4} = Codel{Mapping{1,i}(1,1),2};   
end

Samples = ClinicalData(:,1);
Clinical = ClinicalData(:,2:end);

%% Remove patients with lots of missing features

MissingThr_Pat = 0; %threshold of missing values after which patients are discarded

Clinical1 = Clinical;
j = 0;
for i = 1:length(Clinical(:,1))
   
    dummy = cell2mat(Clinical1(i,:));
    dummy(isnan(dummy)~=1) = 0;
    dummy(isnan(dummy)==1) = 1;
    
    % remove patients with missing survival or censorship status or more
    % than the threshold number of features
    if sum(dummy(1,:)) > (MissingThr_Pat * length(Clinical(1,:))) || ...
            isnan(cell2mat(Clinical1(i,end-1)))==1 || isnan(cell2mat(Clinical1(i,end)))==1
        
        Clinical(i-j,:) = [];
        Samples(i-j,:) = [];
        j = j+1; %since when you delete a feature the index of the "genomics" matrix shifts by one
    end
    
    
end

Survival = cell2mat(Clinical(:,end-1));
Censored = cell2mat(Clinical(:,end));
Clinical = cell2mat(Clinical(:,1:end-2));

%% A bit more preprocessing

% Z-score standardization so that features are comparable
for i = 1:length(Clinical(1,:))
    Mu = mean(Clinical(:,i));
    StDev = std(Clinical(:,i));
    
    Clinical(:,i) = (Clinical(:,i)-Mu)./StDev;
end

% Delete features with NAN values becuase their StDev was 0
Clinical1 = Clinical;
j = 0;
for i = 1:length(Clinical1(1,:))
    
    if isnan(max(Clinical1(:,i))) == 1 
        Clinical(:,i-j) = [];
        Symbols(:,i-j) = [];
        j = j+1; %since when you delete a feature the index of the "mutations" matrix shifts by one
    end
end

% Add 2 to survival to ignore negative values
Survival = Survival+2;

%% Write output structure
clear('ClinicalData')
ClinicalData.Patients = Samples;
ClinicalData.FeatureNames = Symbols;
ClinicalData.Data = Clinical;
ClinicalData.Survival = Survival;
ClinicalData.Censored = Censored;