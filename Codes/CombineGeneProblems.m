function [GeneProblems_Genes,GeneProblems] = CombineGeneProblems(Symbols_mut,Mutations,CopyNo_Genes,CopyNo_Binary)

% This produces a binary table that uses an "OR" operator on mutation and copy# data.
% i.e. for every gene, a patient sample gets a "1" if it EITHER is mutated
% or has a copy number alteration and "0" otherwise.
% 
% inputs:
%
% Symbols_mut - mutation gene names (rows)
% Mutations - mutations data (patients are rows, genes are columns)
% CopyNo_Genes - copy# gene names (rows)
% CopyNo_Binary - copy# data (patients are rows, genes are columns)
%

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

% Getting final combined matrix of gene problems
GeneProblems_Genes = [inCommon_Genes;Symbols_mut;CopyNo_Genes];
GeneProblems = [inCommon;Mutations;CopyNo_Binary];

% sorting genes 
GP = [GeneProblems_Genes,num2cell(GeneProblems)];
GP = sortrows(GP,1);

% returning to patients in rows, genes in columns configuration
GeneProblems_Genes = GP(:,1)';
GeneProblems = GP(:,2:end)';

end