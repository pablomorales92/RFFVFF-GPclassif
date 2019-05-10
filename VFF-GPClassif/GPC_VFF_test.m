function output = GPC_VFF_test(model, Xts, yts)
% Xts: (nts x d) matrix of features for the test set.
% yts: the labels for the test set (in order to compute the accuracy, ROCcurve, AUC...).
% model: The model returned by the 'GPC_VFF_train' function.
W = model.W;
theta = model.theta;
L = model.L;
mu_beta = model.mu_beta;
D = size(W,1);
%% Computing the Random Features for test points
XtsW = Xts * W';
Zts = zeros(size(Xts,1),2*D);
Zts(:,1:2:(2*D)) = (theta(1)/realsqrt(D))*cos(XtsW./theta(2));
Zts(:,2:2:(2*D)) = (theta(1)/realsqrt(D))*sin(XtsW./theta(2));

%% Computing means and variances for f*'s
mean_test = Zts*mu_beta;
aux1 = L\(Zts');
var_test = dot(aux1,aux1)';

%% Calculating Probabilities
probabilities = 1./(1+exp(-(mean_test./realsqrt(1+0.125*pi*var_test))));

%% Calculating predictions
y_predic = probabilities >= 0.5;

%% Measuring results
res = measures(yts,y_predic);
[res.roc.X,res.roc.Y,res.roc.T,res.AUC] = perfcurve(yts,probabilities,'1');
%% Preparing output
output.res = res;
output.prob = probabilities;
output.y_predic = y_predic;

output.means = mean_test;
output.variances = var_test;
end

function results = measures(y_test,y_predic)
%% Only binary case
CM = zeros(2,2);
ind = (y_test==1);

%% Confussion Matrix

    % True Positives
        CM(1,1) = sum(y_predic(ind)==1);
    % True Negatives
        CM(2,2) = sum(y_predic(not(ind))==0);
    % False Positives
        CM(1,2) = sum(y_predic(not(ind))==1);
    % False Negatives
        CM(2,1) = sum(y_predic(ind)==0);

results.CM = CM;

%% Overall Accuracy
    results.OA = 100*(sum(diag(CM)) / sum(sum(CM)));

%% Precision and Recall
    PR(1) = CM(1,1)/(CM(1,1) + CM(1,2));
    PR(2) = CM(1,1)/(CM(1,1) + CM(2,1));
    
    results.Pre_Rec = PR;
%% F-score
    results.Fscore = 2*PR(1)*PR(2)/sum(PR);
    
%% TPR and FPR

    T_F(1) = CM(1,1)/(CM(1,1) + CM(2,1));
    T_F(2) = CM(1,2)/(CM(1,2) + CM(2,2));

    results.TF_ratio = T_F;
end