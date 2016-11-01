LGGAnalysis_Genomic

%% Getting genes in common for mutation and copy# data

% Mapping copy# genes to mutation genes
Mapping2 = StringMatch(Symbols_mut(1,:), CopyNo_Genes(1,:));
% Mapping mutation genes to copy# genes
Mapping3 = StringMatch(CopyNo_Genes(1,:),Symbols_mut(1,:));

% Getting genes in common between mutation and copy# data
inCommon_Genes = Symbols_mut';
inCommon_mut = Mutations';

inCommon_Genes(cellfun(@isempty, Mapping2),:) = [];
inCommon_mut(cellfun(@isempty, Mapping2),:) = [];

inCommon_CopyNo = CopyNo_Binary';
inCommon_CopyNo(cellfun(@isempty, Mapping3),:) = [];

% Combining copy# and mutation 
inCommon = inCommon_mut + inCommon_CopyNo;
inCommon(inCommon>0)=1;

% Removing in-common genes from mutation and copy no data
Symbols_mut = Symbols_mut';
Mutations = Mutations';
Symbols_mut(cellfun(@isNotEmpty, Mapping2),:) = [];
Mutations(cellfun(@isNotEmpty, Mapping2),:) = [];

CopyNo_Genes = CopyNo_Genes';
CopyNo_Binary = CopyNo_Binary';
CopyNo_Genes(cellfun(@isNotEmpty, Mapping3),:) = [];
CopyNo_Binary(cellfun(@isNotEmpty, Mapping3),:) = [];

% Returning to ordinary configuration of rows and columns
inCommon = inCommon';
inCommon_Genes = inCommon_Genes';
Symbols_mut = Symbols_mut';
Mutations = Mutations';
CopyNo_Genes = CopyNo_Genes';
CopyNo_Binary = CopyNo_Binary';

% Getting final combined matrix of gene problems
GeneProblems_Genes = [inCommon_Genes,Symbols_mut,CopyNo_Genes];
GeneProblems = [inCommon,Mutations,CopyNo_Binary];