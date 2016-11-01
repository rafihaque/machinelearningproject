function [CopyNo_Genes,CopyNo_Binary] = BinarizeCopyNo(CopyNo,GeneList,GeneRoles)

%
% This function binarizes gene copy number data in the following manner:
% Oncogene - considered altered if significant amplification
% Tumor Suppressor Gene - ~ ~ if homozygous deletion
%

% % Matrix of gene copy # (SAMPLES IN ROWS, GENES IN COLUMNS)
% CopyNo = Gene_CNV;
% % List of genes to look for in each sample (ROW-BASED)
% GeneList = Symbols_gCNV';
% % Known roles of SOME genes (Oncogene vs Tumor Suppressor Genes)
% GeneRoles = text2cell2('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/VogelsteinGenes_forAnalysis.txt', '\t',1);

%% Getting list of genes with known and unknown roles

% Only keeping relevant genes in the GeneRoles cell array
% sort gene list alphabetically
GeneList_sorted = sort(GeneList);
% map 'gene list' to 'gene roles'
Mapping1 = StringMatch(GeneRoles(:,1), GeneList_sorted);
GeneRoles(cellfun(@isempty, Mapping1),:) = [];

% Getting list of genes with unknown roles
% sorting gene role list (first column)
GeneRoles_sorted = sort(GeneRoles(:,1));
% map 'gene roles' to 'gene list'
GeneList_NoRole = GeneList;
Mapping2 = StringMatch(GeneList_NoRole(:,1), GeneRoles_sorted);
GeneList_NoRole(cellfun(@isNotEmpty, Mapping2),:) = [];

%% Applying final relevant gene list to sample database

% +2 (high-level amplifications only)
CopyNo_Ampl = CopyNo; 
CopyNo_Ampl(CopyNo_Ampl ~= 2) = 0;
CopyNo_Ampl(CopyNo_Ampl == 2) = 1;
% -2 (homozygous -total- deletions only)
CopyNo_Del = CopyNo; 
CopyNo_Del(CopyNo_Del ~= (-2)) = 0;
CopyNo_Del(CopyNo_Del == (-2)) = 1;

%% Determining gene roles of unknown genes

sum_Ampl = sum(CopyNo_Ampl); %total amplifications
sum_Del = sum(CopyNo_Del); %total deletions
s_ampl = num2cell(sum_Ampl); %converting to cell array
s_del = num2cell(sum_Del); %converting to cell array

g_norole = [GeneList,s_ampl',s_del'];

% sorting  Norole list (first column)
NoRole_sorted = sort(GeneList_NoRole(:,1));
% map 'NoRole' to 'g_norol'
Mapping4 = StringMatch(g_norole(:,1), NoRole_sorted);
g_norole(cellfun(@isempty, Mapping4),:) = [];

% Making a decision (+2 = oncogene ; -2 = TSG)
nD = cell2mat(g_norole(:,2:end));
noroleDecision = nD(:,1) - nD(:,2);
noroleDecision(noroleDecision>=0) = 2;
noroleDecision(noroleDecision<0) = -2;
% Updating each unknown gene with its decision
noroleDecision = num2cell(noroleDecision);
g_norole = [g_norole(:,1),noroleDecision];

% concatenating roles of ALL genes
for i = 1:length(GeneRoles(:,2)) 
    GeneRoles{i,2} = str2num(GeneRoles{i,2});
end
GeneRoles = [GeneRoles;g_norole];

%% Applying binarization decisions to dataset

% Sorting genes alphabetically
GeneRoles = sortrows(GeneRoles);
% Original list of genes and alterations
Contents = [GeneList,num2cell(CopyNo')];
Contents = sortrows(Contents); %sorting to correspond with gene roles list

% Getting updated Copy number and g roles matrices
CopyNo = cell2mat(Contents(:,2:end));
groles = cell2mat(GeneRoles(:,2));

% Considering any oncogene with a +2 (high-level amplification) and any
% tumor suppressor gene with a -2 (homozygous deletion) to be a copy number
% alteration in tumor sample

for i = 1:length(groles(:,1)) 
    
    if groles(i,1) == 2 %Oncogene
        c = CopyNo(i,:);
        c(c~=2) = 0;
        c(c==2) = 1;
        CopyNo(i,:) = c;
        
    elseif groles(i,1) == (-2) %Tumor Suppressor gene
        c = CopyNo(i,:);
        c(c~=(-2)) = 0;
        c(c==(-2)) = 1;
        CopyNo(i,:) = c;
    
    end
end

CopyNo_Binary = CopyNo';
CopyNo_Genes = Contents(:,1)';

end