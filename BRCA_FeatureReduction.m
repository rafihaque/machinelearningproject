clear; close all; clc;
 
% add relevant paths
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Data/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/old/')
addpath('/home/mohamed/Desktop/Class/CS534-MachineLearning/Class Project/Codes/glmnet_matlab/glmnet_matlab/')


%% Read in data and initial preprocessing and feature extraction
 
load 'BRCA.Data.mat';
 
% convert to better format
SymbolTypes = cellstr(SymbolTypes);
Symbols = cellstr(Symbols);
Samples = (cellstr(Samples))';
 
% Remove features with zero variance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nonNAN_sum = Features;
nonNAN_sum(isnan(nonNAN_sum)==1) = 0;
nonNAN_sum = sum(nonNAN_sum, 2);
nonNAN_count = ~isnan(Features);
nonNAN_count = sum(nonNAN_count, 2);
nonNAN_mean = nonNAN_sum ./ nonNAN_count;
[~,nonNAN_mean] = meshgrid(1:length(Features(1,:)), nonNAN_mean);
 
Features_mean = nonNAN_mean;
Features_var = (Features - nonNAN_mean) .^ 2;
 
nonNAN_sum = Features_var;
nonNAN_sum(isnan(nonNAN_sum)==1) = 0;
nonNAN_sum = sum(nonNAN_sum, 2);
nonNAN_count = ~isnan(Features_var);
nonNAN_count = sum(nonNAN_count, 2);
Features_var = nonNAN_sum ./ nonNAN_count;
 
Features(Features_var == 0, :) = [];
Symbols(Features_var == 0, :) = [];
SymbolTypes(Features_var == 0, :) = [];
 
Features_mean(Features_var == 0, :) = [];
Features_var(Features_var == 0, :) = [];
 
% Z- score standardization of features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,Features_var] = meshgrid(1:length(Features(1,:)), Features_var);
Features = (Features - Features_mean) ./ (Features_var .^ 0.5);
 
% define different feature matrices %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Clinical = Features(strcmp(SymbolTypes,'Clinical'), :);
Mutation = Features(strcmp(SymbolTypes, 'Mutation'), :);
CNVGene = Features(strcmp(SymbolTypes, 'CNVGene'), :);
CNVArm = Features(strcmp(SymbolTypes, 'CNVArm'), :);
Protein = Features(strcmp(SymbolTypes, 'Protein'), :);
mRNA = Features(strcmp(SymbolTypes, 'mRNA'), :);
 
% save original for monitoring what happens as you delete stuff %%%%%%%%%%%
SymbolTypes_original = SymbolTypes;
Symbols_original = Symbols;
Features_original = Features;
Clinical_original = Clinical;
Mutation_original = Mutation;
CNVGene_original = CNVGene;
CNVArm_original = CNVArm;
Protein_original = Protein;
mRNA_original = mRNA;
 
%% Make decisions
 
% Delete any patient or feature missing a single value?
Delete_Only = 0;
 
% Decide which feature sets to process
Process_Clinical = 1;
Process_Mutation = 1;
Process_CNVGene = 1;
Process_CNVArm = 1;
Process_Protein = 1;
Process_mRNA = 1;
 
% Decide which features to impute
Impute_Clinical = 0;
Impute_Mutation = 0;
Impute_CNVGene = 0;
Impute_CNVArm = 0;
Impute_Protein = 0;
Impute_mRNA = 0;

%% Define thresholds and parameters
 
% Define thresholds (for patient removal)
Thr_clin_p = 0;
Thr_mut_p = 0.7;
Thr_cnvgene_p = 0.1;
Thr_cnvarm_p = 1;
Thr_protein_p = 0.5;
Thr_mrna_p = 0.35;

% Define thresholds (for features removal)
Thr_clin_f = 0.01;
Thr_mut_f = 0.3;
Thr_cnvgene_f = 1;
Thr_cnvarm_f = 0.1;
Thr_protein_f = 0.2;
Thr_mrna_f = 1;
 
% Define imputation parameters
K_impute = 30;
K_mode = 'Regression';
 
 
%% CLINICAL - Remove patients and features accordingly (order matters!)
 
if Process_Clinical == 1
 
% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clin_f = sum((isnan(Clinical)==1),2);
 
Clinical (clin_f > (Thr_clin_f * length(Clinical(1,:))), :) = [];
 
Symbols (clin_f > (Thr_clin_f * length(Clinical(1,:))), :) = [];
SymbolTypes (clin_f > (Thr_clin_f * length(Clinical(1,:))), :) = [];
 
% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clin_p = sum((isnan(Clinical)==1),1);
 
Clinical (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Mutation (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
CNVGene (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
CNVArm (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Protein (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
mRNA (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Survival (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Censored (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
Samples (:, clin_p > (Thr_clin_p * length(Clinical(:,1)))) = [];
 
% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_Clinical == 1
Clinical = KNN_Impute(Clinical,K_impute,K_mode);
end
 
end
 
%% MUTATION - Remove patients and features accordingly (order matters!)
 
if Process_Mutation == 1
 
% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mut_f = sum((isnan(Mutation)==1),2);
 
Mutation (mut_f > (Thr_mut_f * length(Mutation(1,:))), :) = [];
 
feat_Sofar = length(Clinical(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; mut_f > (Thr_mut_f * length(Mutation(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];
 
% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mut_p = sum((isnan(Mutation)==1),1);
 
Clinical (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Mutation (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
CNVGene (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
CNVArm (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Protein (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
mRNA (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Survival (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Censored (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
Samples (:, mut_p > (Thr_mut_p * length(Mutation(:,1)))) = [];
 
% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_Mutation == 1
Mutation = KNN_Impute(Mutation,K_impute,K_mode);
end
 
end
 
%% CNVGene - Remove patients and features accordingly (order matters!)
 
if Process_CNVGene == 1
 
% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvgene_f = sum((isnan(CNVGene)==1),2);
 
CNVGene (cnvgene_f > (Thr_cnvgene_f * length(CNVGene(1,:))), :) = [];
 
feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; cnvgene_f > (Thr_cnvgene_f * length(CNVGene(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];
 
% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvgene_p = sum((isnan(CNVGene)==1),1);
 
Clinical (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Mutation (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
CNVGene (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
CNVArm (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Protein (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
mRNA (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Survival (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Censored (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
Samples (:, cnvgene_p > (Thr_cnvgene_p * length(CNVGene(:,1)))) = [];
 
% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_CNVGene == 1
CNVGene = KNN_Impute(CNVGene,K_impute,K_mode);
end
 
end
 
%% CNVArm - Remove patients and features accordingly (order matters!)
 
if Process_CNVArm == 1
 
% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvarm_f = sum((isnan(CNVArm)==1),2);
 
CNVArm (cnvarm_f > (Thr_cnvarm_f * length(CNVArm(1,:))), :) = [];
 
feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1)) + length(CNVGene(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; cnvarm_f > (Thr_cnvarm_f * length(CNVArm(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];
 
% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cnvarm_p = sum((isnan(CNVArm)==1),1);
 
Clinical (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Mutation (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
CNVGene (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
CNVArm (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Protein (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
mRNA (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Survival (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Censored (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
Samples (:, cnvarm_p > (Thr_cnvarm_p * length(CNVArm(:,1)))) = [];
 
% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_CNVArm == 1
CNVArm = KNN_Impute(CNVArm,K_impute,K_mode);
end
 
end
 
%% Protein - Remove patients and features accordingly (order matters!)
 
if Process_Protein == 1 
 
% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
protein_f = sum((isnan(Protein)==1),2);
 
Protein (protein_f > (Thr_protein_f * length(Protein(1,:))), :) = [];
 
feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1)) + length(CNVGene(:,1)) + length(CNVArm(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; protein_f > (Thr_protein_f * length(Protein(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];
 
% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
protein_p = sum((isnan(Protein)==1),1);
 
Clinical (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Mutation (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
CNVGene (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
CNVArm (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Protein (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
mRNA (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Survival (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Censored (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = [];
Samples (:, protein_p > (Thr_protein_p * length(Protein(:,1)))) = []; 
 
% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_Protein == 1
Protein = KNN_Impute(Protein,K_impute,K_mode);
end
 
end
 
%% mRNA - Remove patients and features accordingly (order matters!)
 
if Process_mRNA == 1
 
% remove FEATURES missing too many patients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mrna_f = sum((isnan(mRNA)==1),2);
 
mRNA (mrna_f > (Thr_mrna_f * length(mRNA(1,:))), :) = [];
 
feat_Sofar = length(Clinical(:,1)) + length(Mutation(:,1)) + length(CNVGene(:,1)) + length(CNVArm(:,1)) + length(Protein(:,1));
feat_Sofar = [zeros(feat_Sofar,1) ; mrna_f > (Thr_mrna_f * length(mRNA(1,:)))];

Symbols (feat_Sofar == 1, :) = [];
SymbolTypes (feat_Sofar == 1, :) = [];
 
% remove PATIENTS missing too many features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mrna_p = sum((isnan(mRNA)==1),1);
 
Clinical (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Mutation (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
CNVGene (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
CNVArm (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Protein (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
mRNA (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Survival (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Censored (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
Samples (:, mrna_p > (Thr_mrna_p * length(mRNA(:,1)))) = [];
 
% IMPUTING missing values %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if Impute_mRNA == 1
mRNA = KNN_Impute(mRNA,K_impute,K_mode);
end
 
end
 
%% Updating final features matrix
 
Features = [Clinical; Mutation; CNVGene; CNVArm; Protein; mRNA];
 
%% SURVIVAL AND CENSORED - Remove patients missing survival/censored data
 
Features(:,isnan(Survival)==1) = [];
Censored(:,isnan(Survival)==1) = [];
Samples(:,isnan(Survival)==1) = [];
Survival(:,isnan(Survival)==1) = [];
 
Features(:,isnan(Censored)==1) = [];
Survival(:,isnan(Censored)==1) = [];
Samples(:,isnan(Censored)==1) = [];
Censored(:,isnan(Censored)==1) = [];

%% Pack data

ReducedModel.Features = Features;
ReducedModel.Symbols = Symbols;
ReducedModel.SymbolTypes = SymbolTypes; 
ReducedModel.Survival = Survival;
ReducedModel.Censored = Censored;
ReducedModel.Samples = Samples;
