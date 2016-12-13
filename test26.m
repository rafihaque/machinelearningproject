Features_NonCodel = a;

% Z-score standardization
Features_mean_NonCodel = mean(Features_NonCodel, 2);
[~,Features_mean_NonCodel] = meshgrid(1:length(Features_NonCodel(1,:)), Features_mean_NonCodel);
Features_std_NonCodel = std(Features_NonCodel');
[~,Features_std_NonCodel] = meshgrid(1:length(Features_NonCodel(1,:)), Features_std_NonCodel');
Features_Zscored_NonCodel = (Features_NonCodel - Features_mean_NonCodel) ./ Features_std_NonCodel;
Features_Zscored_NonCodel(isnan(Features_Zscored_NonCodel)==1) = 0;