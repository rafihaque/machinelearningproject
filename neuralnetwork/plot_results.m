load('~/machinelearningproject/neuralnetwork/Results5.mat')


for i = 1:size(ypred,1)
  for j = 1:size(ypred,2)
    [~,~,~,auc(i,j)] = perfcurve(squeeze(ytest(i,j,:)),squeeze(ypred(i,j,:)),1)
  end
end
boxplot(auc)