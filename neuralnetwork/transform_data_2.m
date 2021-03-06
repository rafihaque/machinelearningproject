function [newdata] = transform_data_2(data)

% convert time to years
survival = round(data.Survival/365)+1;
features = data.Features;

censored  = logical(data.Censored);
dead      = nan(sum(survival),1);
tfeatures = nan(sum(survival),size(features,1));



% loop through features to create new matrix
counter = 0;
counter2 = 0
dead = []
tf = []
sfeat = []
for i = 1:size(features,2)
  if censored(i)
    tf = [tf repmat(features(:,i),1,survival(i))];
    dead = [dead repmat(0,1,survival(i))];
    sfeat = [sfeat 1:survival(i)];
  else
    tf = [tf repmat(features(:,i),1,2*survival(i))];
    dead = [dead repmat(0,1,survival(i))];
    dead = [dead repmat(1,1,survival(i))];
    sfeat = [sfeat 1:2*survival(i)];
    
  end
end

  



newdata.features = zscore([tf; sfeat]');
newdata.features = [tf; sfeat]';
newdata.survival = [dead; ~dead]';

save('~/machinelearningproject/neuralnetwork/transformed_data_2','newdata')


