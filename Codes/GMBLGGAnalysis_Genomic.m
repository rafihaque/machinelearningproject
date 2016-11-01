%
% This fetches the most relevant genomic features
% from GBMLGG data
%

clear ; close all ; clc ; 

%% Load LGG data

load('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/GBMLGG.Data.mat')
Symbols = cellstr(Symbols);
Samples = cellstr(Samples);

%% Extract genomic features
Idx_Start = 14; %start of genomic features
Idx_End = 229; %end of genomic features

CNV_Start = 142; %beginning of CNV data
Chrom_Start = 205; %beginning of chromosomal copy# data

Genomic = Features(Idx_Start:Idx_End,:)';
Symbols = Symbols(Idx_Start:Idx_End,:)';

% removing extra "postfixes" to feature names
for i = 1:length(Symbols(1,:))
featName = strsplit(char(Symbols(1,i)),'_');
Symbols(1,i) = featName(1,1);
end

SymbolTypes = zeros(1,length(SymbolTypes));
SymbolTypes(1,Idx_Start:CNV_Start-1) = 1; %mutation data
SymbolTypes(1,CNV_Start:Chrom_Start-1) = 2; %gene CNV data
SymbolTypes(1,Chrom_Start:Idx_End) = 3; %chromosomal CNV data
SymbolTypes = SymbolTypes(:,Idx_Start:Idx_End);

Survival = Survival';
Censored = Censored';

Genomic = [Genomic,Survival,Censored];

%% Remove patients with lots of missing features

MissingThr_Pat = 0.2; %threshold of missing values after which patients are discarded

Genomic1 = Genomic;
j = 0;
for i = 1:length(Genomic(:,1))
   
    dummy = Genomic1(i,:);
    dummy(isnan(dummy)~=1) = 0;
    dummy(isnan(dummy)==1) = 1;
    
    % remove patients with missing survival or censorship status or more
    % than the threshold number of features
    if sum(dummy(1,:)) > (MissingThr_Pat * length(Genomic(1,:))) || ...
            isnan(Genomic1(i,end-1))==1 || isnan(Genomic1(i,end))==1
        
        Genomic(i-j,:) = [];
        Samples(i-j,:) = [];
        j = j+1; %since when you delete a feature the index of the "genomics" matrix shifts by one
    end
    
    
end


%% Remove features with lots of missing data

MissingThr_Feat = 0.2; %threshold of missing values after which features are discarded

Genomic1 = Genomic;
j = 0;
for i = 1:length(Genomic(1,:)) -2 %-2 to ignore survival and censorship
    
    dummy = Genomic1(:,i);
    dummy(isnan(dummy)~=1) = 0;
    dummy(isnan(dummy)==1) = 1;
    
    if sum(dummy(:,1)) > (MissingThr_Feat * length(Genomic(:,1)))
        Genomic(:,i-j) = [];
        Symbols(:,i-j) = [];
        SymbolTypes(:,i-j) = [];
        j = j+1; %since when you delete a feature the index of the "genomics" matrix shifts by one
    end
    
end

Survival = Genomic(:,end-1);
Censored = Genomic(:,end);
Genomic = Genomic(:,1:end-2);

%% Separate mutations, gene CNV and chromosomal CNV data

Mutations = Genomic;
Gene_CNV = Genomic;
Chrom_CNV = Genomic;

Symbols_mut = Symbols;
Symbols_gCNV = Symbols;
Symbols_cCNV = Symbols;

j_m = 0;
j_g = 0;
j_c = 0;
for i = 1:length(Genomic(1,:))
    
    if SymbolTypes(1,i) ~= 1
        Mutations(:,i-j_m) = [];
        Symbols_mut(:,i-j_m) = [];
        j_m = j_m+1; %since when you delete a feature the index of the "mutations" matrix shifts by one
    end
    if SymbolTypes(1,i) ~= 2
        Gene_CNV(:,i-j_g) = [];
        Symbols_gCNV(:,i-j_g) = [];
        j_g = j_g+1;
    end
    if SymbolTypes(1,i) ~= 3
        Chrom_CNV(:,i-j_c) = [];
        Symbols_cCNV(:,i-j_c) = [];
        j_c = j_c+1; 
    end
end

%% Binarizing gene copy # data

% Known roles of SOME genes (Oncogene vs Tumor Suppressor Genes)
GeneRoles = text2cell2('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/VogelsteinGenes_forAnalysis.txt', '\t',1);

[CopyNo_Genes,CopyNo_Binary] = BinarizeCopyNo(Gene_CNV,Symbols_gCNV',GeneRoles);

%% Combine gene problems 

[GeneProblems_Genes,GeneProblems] = CombineGeneProblems(Symbols_mut,Mutations,CopyNo_Genes,CopyNo_Binary);

GeneProblems = cell2mat(GeneProblems);

%% A bit more preprocessing

% Z-score standardization so that features are comparable
for i = 1:length(GeneProblems(1,:))
    Mu = mean(GeneProblems(:,i));
    StDev = std(GeneProblems(:,i));
   
    GeneProblems(:,i) = (GeneProblems(:,i)-Mu)./StDev;
end

% Delete features with NAN values becuase their StDev was 0
GeneProblems1 = GeneProblems;
j = 0;
for i = 1:length(GeneProblems1(1,:))
    
    if isnan(max(GeneProblems1(:,i))) == 1 
        GeneProblems(:,i-j) = [];
        GeneProblems_Genes(:,i-j) = [];
        j = j+1; %since when you delete a feature the index of the "mutations" matrix shifts by one
    end
end

% Add 2 to survival to ignore negative values
Survival = Survival+2;


%% Write output structure

GenomicData.Patients = Samples;
GenomicData.FeatureNames = GeneProblems_Genes;
GenomicData.Data = GeneProblems;
GenomicData.Survival = Survival;
GenomicData.Censored = Censored;