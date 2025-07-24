% fslssvm_ex2_2.m - Fixed-size LS-SVM for Shuttle and California datasets
clc; clear; close all;
addpath('../LSSVMlab');

% Uncomment one of the following to select your dataset:
% dataset = 'shuttle';
dataset = 'shuttle';


switch dataset
    case 'shuttle'
        %% 2.2.1 Shuttle (classification)
        data = load('shuttle.dat','-ascii');
        data = data(1:1000,:);  % use first 700 samples for faster experimentation
        function_type = 'c';  % 'c' for classification
        X = data(:,1:end-1);
        Y = data(:,end);
        
        % Explore dataset
        fprintf('Shuttle dataset: %d samples, %d features\n', size(X,1), size(X,2));
        classes = unique(Y);
        fprintf('Number of classes: %d (labels %s)\n', numel(classes), mat2str(classes'));
        
        figure; histogram(Y);
        title('Shuttle: Class distribution');
        
        %model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
        % Tune hyperparameters using standard cross-validation
        %[gam, sig2] = tunelssvm(model, 'simplex', 'crossvalidatelssvm', {5, 'mse'});
        
        %[ selected , ranking ] = bay_lssvmARD ({ X , Y , 'f', gam , sig2 }) ;
        
    case 'california'
        %% 2.2.2 California (regression)
        data = load('california.dat','-ascii');
        data = data(1:2500,:);  % use first 2500 samples for faster experimentation
        function_type = 'f';  % 'f' for regression
        X = data(:,1:end-1);
        Y = data(:,end);
        
        %model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');
        % Tune hyperparameters using standard cross-validation
        %[gam, sig2] = tunelssvm(model, 'simplex', 'crossvalidatelssvm', {5, 'mse'});
        
        %[ selected , ranking ] = bay_lssvmARD ({ X , Y , 'f', gam , sig2 }) ;

        % Explore dataset
        fprintf('California housing: %d samples, %d features\n', size(X,1), size(X,2));
        
        figure; histogram(Y);
        title('California: Median house value distribution');
        xlabel('Median house value'); ylabel('Frequency');
        
        % Visualize location vs house value (Longitude = col 5, Latitude = col 6)
        figure; scatter(X(:,5), X(:,6), 10, Y, 'filled');
        colorbar; xlabel('Longitude'); ylabel('Latitude');
        title('House value by geographic location');
end


% Split into training and test sets (80/20 split)
cv = cvpartition(Y, 'HoldOut', 0.25, 'Stratify', true);
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
Xtest  = X(test(cv),:);
Ytest  = Y(test(cv),:);
% FS-LSSVM parameters
k = 4;                             % for fsoperations
kernel_type = 'RBF_kernel';        % 'lin_kernel' or 'rbf_kernel'
global_opt  = 'csa';               % 'csa' or 'ds'
user_process = {'FS-LSSVM','SV_L0_norm'};
window = [15,20,25];               % representer budgets to test

% Run Fixed-size LS-SVM over different budgets
[e, s, t] = fslssvm(Xtrain, Ytrain, k, function_type, kernel_type, global_opt, user_process, window, Xtest, Ytest);

% 3) Compute averages across the 10 runs:
avgErrCount  = mean(e,  2);   % mean number of misclassifications
avgSV        = mean(s,  2);   % mean # of support vectors
avgTime      = mean(t,  2);   % mean CPU time (seconds)

% 4) Convert avgErrCount to percent accuracy:
numTest = numel(Ytest);
avgAccuracy = (1 - (avgErrCount / (numTest/10))) * 100;

% 5) Print everything neatly:
for i = 1:length(user_process)
    fprintf('%s (k=%d):\n', user_process{i}, k);
    fprintf('  Avg. Misclassified : %.2f samples\n', avgErrCount(i));
    fprintf('  Avg. Accuracy      : %.2f%%\n',    avgAccuracy(i));
    fprintf('  Avg. # SV          : %.1f\n',      avgSV(i));
    fprintf('  Avg. Time (sec)    : %.2f\n\n',    avgTime(i));
end