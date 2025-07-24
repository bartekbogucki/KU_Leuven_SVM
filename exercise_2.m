X = (-3:0.01:3)';
Y = sinc(X) + 0.1.*randn(length(X), 1);

% Split data
Xtrain = X(1:2:end);
Ytrain = Y(1:2:end);
Xtest = X(2:2:end);
Ytest = Y(2:2:end);

% Hyperparameter ranges
%gam_values = [10, 10.^3, 10.^6];
%sig2_values = [0.01, 1, 100];

% Store results
%mse_results = zeros(length(gam_values), length(sig2_values));

% plot_idx = 1;
% for i = 1:length(gam_values)
%     for j = 1:length(sig2_values)
%         gam = gam_values(i);
%         sig2 = sig2_values(j);
% 
%         % Train LS-SVM
%         [alpha,b] = trainlssvm({Xtrain,Ytrain,'function estimation',gam,sig2,'RBF_kernel','preprocess'});
% 
%         % Predict
%         Ypred = simlssvm({Xtrain,Ytrain,'function estimation',gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);
% 
%         % Compute MSE
%         mse = mean((Ypred - Ytest).^2);
%         mse_results(i,j) = mse;
%         fprintf('gam = %.0e, sig2 = %.2f -> MSE = %.4f\n', gam, sig2, mse);
% 
%         % Plot
%         figure;
%         plotlssvm({Xtrain,Ytrain,'function estimation',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
%         hold on;
%         plot(X, sinc(X), 'r-.');
%         scatter(Xtest, Ytest, 10, 'k', 'filled');
%         title(sprintf('gam=%.0e, sig2=%.2f', gam, sig2));
%         plot_idx = plot_idx + 1;
%         legend('LS-SVM Prediction', 'Train data', 'True sinc', 'Test data', 'Location', 'northeast');
%         hold off;
%     end
% end
% 

type = 'function estimation';
[gam,sig2] = tunelssvm({X,Y,type,[],[],'RBF_kernel'},'gridsearch','leaveoneoutlssvm',{'mse'});

% Train LS-SVM
[alpha,b] = trainlssvm({Xtrain,Ytrain,'function estimation',gam,sig2,'RBF_kernel','preprocess'});

% Predict
Ypred = simlssvm({Xtrain,Ytrain,'function estimation',gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xtest);

% Compute MSE
mse = mean((Ypred - Ytest).^2);
mse_results(i,j) = mse;
fprintf('gam = %.0e, sig2 = %.2f -> MSE = %.4f\n', gam, sig2, mse);

% Plot
figure;
plotlssvm({Xtrain,Ytrain,'function estimation',gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
hold on;
plot(X, sinc(X), 'r-.');
scatter(Xtest, Ytest, 10, 'k', 'filled');
title(sprintf('gam=%.2f, sig2=%.2f', gam, sig2));
plot_idx = plot_idx + 1;
legend('LS-SVM Prediction', 'Train data', 'True sinc', 'Test data', 'Location', 'northeast');
hold off;

%1.2.2

sig2 = 0.4;
gam = 10;
crit_L1 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;
crit_L2 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
crit_L3 = bay_lssvm ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;
[~ , alpha , b ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 1) ;
[~ , gam ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 2) ;
[~ , sig2 ] = bay_optimize ({ Xtrain , Ytrain , 'f', gam , sig2 } , 3) ;
sig2e = bay_errorbar ({ Xtrain , Ytrain , 'f', gam , sig2 } , 'figure') ;


%1.3
X = 6.* rand (100 , 3) - 3;
Y = sinc ( X (: ,1) ) + 0.1.* randn (100 ,1) ;
[ selected , ranking ] = bay_lssvmARD ({ X , Y , 'f', gam , sig2 }) ;

% Horizontal bar plot
figure;
barh(ranking, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'YTick', 1:size(X,2), 'YDir', 'reverse');  % Reverse Y-axis to show most relevant at top
xticks(1:length(ranking));                         % X-axis ticks at each integer
xlim([0 length(ranking)+0.5]);                     % Give some spacing
xlabel('Ranking');
ylabel('Feature column');
title('Input Relevance via Bayesian ARD (Top = Most Relevant)');

% Cross-validation: cumulative feature inclusion
inputs = 1:3;
mse_cv = zeros(1, length(inputs));
type = 'function estimation';

for i = 1:length(inputs)
    Xi = X(:, 1:i);  % use first i features
    mse_cv(i) = crossvalidate({Xi, Y, type, gam, sig2, 'RBF_kernel'}, 10, 'mse');
end

% Horizontal bar plot
figure;
barh(mse_cv, 'FaceColor', [0.5 0.7 0.5]);
set(gca, 'YTick', 1:length(inputs), 'YTickLabel', {'1', '1-2', '1-2-3'}, 'YDir', 'reverse');
xlabel('10-Fold Cross-Validation MSE');
ylabel('Features Used');
title('Effect of forward adding features on Model Performance');

% Cross-validation: backward feature elimination
num_features = size(X, 2);
type = 'function estimation';
mse_backward = zeros(1, num_features);

for i = 1:num_features
    subset = 1:(num_features - i + 1);  % progressively remove last feature
    Xi = X(:, subset);
    mse_backward(i) = crossvalidate({Xi, Y, type, gam, sig2, 'RBF_kernel'}, 10, 'mse');
end
% Prepare labels for the feature subsets
labels = {'1-2-3', '1-2', '1'};

% Horizontal bar plot
figure;
barh(mse_backward, 'FaceColor', [0.5 0.7 0.5]);
set(gca, 'YTick', 1:num_features, 'YTickLabel', labels, 'YDir', 'reverse');
xlabel('10-Fold Cross-Validation MSE');
ylabel('Features Used');
title('Backward Feature Elimination via Cross-Validation');

%1.4
% Generate input data
X = (-6:0.2:6)';
Y = sinc(X) + 0.1 * rand(size(X));  % Add uniform noise

% Add artificial outliers
out = [15 17 19];
Y(out) = 0.7 + 0.3 * rand(size(out));
out = [41 44 46];
Y(out) = 1.5 + 0.2 * rand(size(out));

% Non-robust!
% Initialize model
model = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');

% Tune hyperparameters using standard cross-validation
model = tunelssvm(model, 'simplex', 'crossvalidatelssvm', {10, 'mse'});

% Train model (optional, for consistency)
model = trainlssvm(model);

% Plot result
plotlssvm(model);
title('Non-Robust LS-SVM Regression - no weighting (MSE loss)');

% Robust!
% Initialize model
model_robust = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');

% Tune using robust cross-validation and MAE cost
model_robust = tunelssvm(model_robust, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, 'whuber');

% Train robust model
model_robust = robustlssvm(model_robust);

% Plot result
figure;
plotlssvm(model_robust);
title('Robust LS-SVM Regression - Huber Weight (MAE loss)');

%other - ’whampel’
model_robust = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');

% Tune using robust cross-validation and MAE cost
model_robust = tunelssvm(model_robust, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, 'whampel');

% Train robust model
model_robust = robustlssvm(model_robust);

% Plot result
figure;
plotlssvm(model_robust);
title('Robust LS-SVM Regression - Hampel Weight (MAE loss)');

%other - ’’wlogistic’’
model_robust = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');

% Tune using robust cross-validation and MAE cost
model_robust = tunelssvm(model_robust, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, 'wlogistic');

% Train robust model
model_robust = robustlssvm(model_robust);

% Plot result
figure;
plotlssvm(model_robust);
title('Robust LS-SVM Regression - Logistic Weight (MAE loss)');

%other - ’’’wmyriad’’’
model_robust = initlssvm(X, Y, 'f', [], [], 'RBF_kernel');

% Tune using robust cross-validation and MAE cost
model_robust = tunelssvm(model_robust, 'simplex', 'rcrossvalidatelssvm', {10, 'mae'}, 'wmyriad');

% Train robust model
model_robust = robustlssvm(model_robust);

% Plot result
figure;
plotlssvm(model_robust);
title('Robust LS-SVM Regression - Myriad Weight (MAE loss)');

%2.2
%initial
% Load the data
load logmap.mat  % provides Z (train), Ztest (test)

% Define initial model order
order = 10;

% Convert to autoregressive window format
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);

% Set initial parameters
gam = 10;
sig2 = 10;

% Train model
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});

% Prepare for prediction
Xs = Z(end - order + 1:end, 1);  % last "order" points of training data
nb = length(Ztest);           % number of points to predict

% Predict
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);
mse_test = mean((Ztest - prediction).^2);

% Visualize
figure; hold on;
plot(Ztest, 'k');             % actual future values
plot(prediction, 'r');        % predicted values
legend('Actual', 'Prediction');
title(sprintf('LS-SVM RBF-kernel (order=%d, gam=%.1f, sig2=%.1f)', order, gam, sig2));
subtitle(sprintf('MSE test=%.4f', mse_test));
hold off;

%tuning order gamma sig2
orders = 20:30;               % Try a few model orders

best_mse = Inf;

for ord = orders
    X = windowize(Z, 1:(ord + 1));
    Y = X(:, end);
    X = X(:, 1:ord);

    [gam, sig2, cost] = tunelssvm({X, Y, 'function estimation',[],[],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'mse'});
    
    mse_val = crossvalidate({X, Y, 'function estimation', gam, sig2, 'RBF_kernel'}, 10, 'mse');

    if mse_val < best_mse
        best_mse = mse_val;
        best_order = ord;
        best_gam = gam;
        best_sig2 = sig2;
    end
end

fprintf('Best order: %d, gam: %.4f, sig2: %.4f, MSE: %.4f\n', best_order, best_gam, best_sig2, best_mse);

%with best

% Define initial model order
order = best_order;

% Convert to autoregressive window format
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);

% Set initial parameters
gam = best_gam;
sig2 = best_sig2;

% Train model
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});
mse_val = crossvalidate({X, Y, 'function estimation', gam, sig2, 'RBF_kernel'}, ...
                          10, 'mse');

% Prepare for prediction
Xs = Z(end - order + 1:end, 1);  % last "order" points of training data
nb = length(Ztest);           % number of points to predict

% Predict
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);
mse_test = mean((Ztest - prediction).^2);

% Visualize
figure; hold on;
plot(Ztest, 'k');             % actual future values
plot(prediction, 'r');        % predicted values
legend('Actual', 'Prediction');
title(sprintf('LS-SVM RBF-kernel (order=%d, gam=%.1f, sig2=%.1f)', order, gam, sig2));
subtitle(sprintf('MSE cross-validation=%.4f, MSE test=%.4f', mse_val, mse_test));
hold off;

%2.3
%2.2
%initial
% Load the data
load santafe.mat  % provides Z (train), Ztest (test)

% Define initial model order
order = 50;

% Convert to autoregressive window format
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);

% Set initial parameters

[gam, sig2, cost] = tunelssvm({X, Y, 'function estimation',10,[],'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'mse'});
mse_val = cost;

% Train model
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});

% Prepare for prediction
Xs = Z(end - order + 1:end, 1);  % last "order" points of training data
nb = length(Ztest);           % number of points to predict

% Predict
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);
mse_test = mean((Ztest - prediction).^2);

% Visualize
figure; hold on;
plot(Ztest, 'k');             % actual future values
plot(prediction, 'r');        % predicted values
legend('Actual', 'Prediction');
title(sprintf('LS-SVM RBF-kernel (order=%d, gam=%.1f, sig2=%.1f)', order, gam, sig2));
subtitle(sprintf('MSE cross-validation=%.4f, MSE test=%.4f', mse_val, mse_test));
hold off;

%tuning order gamma sig2
orders = 20:25;               % Try a few model orders

best_mse = Inf;

for ord = orders
    X = windowize(Z, 1:(ord + 1));
    Y = X(:, end);
    X = X(:, 1:ord);

    [gam, sig2, cost] = tunelssvm({X, Y, 'function estimation',10, 50,'RBF_kernel'}, 'simplex', 'crossvalidatelssvm', {10, 'mse'});
    
    mse_val = crossvalidate({X, Y, 'function estimation', gam, sig2, 'RBF_kernel'}, 10, 'mse');

    if mse_val < best_mse
        best_mse = mse_val;
        best_order = ord;
        best_gam = gam;
        best_sig2 = sig2;
    end
end

fprintf('Best order: %d, gam: %.4f, sig2: %.4f, MSE: %.4f\n', best_order, best_gam, best_sig2, best_mse);

%with best

% Define initial model order
order = best_order;

% Convert to autoregressive window format
X = windowize(Z, 1:(order + 1));
Y = X(:, end);
X = X(:, 1:order);

% Set initial parameters
gam = best_gam;
sig2 = best_sig2;

% Train model
[alpha, b] = trainlssvm({X, Y, 'f', gam, sig2});
mse_val = crossvalidate({X, Y, 'function estimation', gam, sig2, 'RBF_kernel'}, ...
                          10, 'mse');

% Prepare for prediction
Xs = Z(end - order + 1:end, 1);  % last "order" points of training data
nb = length(Ztest);           % number of points to predict

% Predict
prediction = predict({X, Y, 'f', gam, sig2}, Xs, nb);
mse_test = mean((Ztest - prediction).^2);

% Visualize
figure; hold on;
plot(Ztest, 'k');             % actual future values
plot(prediction, 'r');        % predicted values
legend('Actual', 'Prediction');
title(sprintf('LS-SVM RBF-kernel (order=%d, gam=%.1f, sig2=%.1f)', order, gam, sig2));
subtitle(sprintf('MSE cross-validation=%.4f, MSE test=%.4f', mse_val, mse_test));
hold off;





% Expanding Window Cross-Validation for Time Series Prediction
% Using LS-SVM on Logmap dataset

% Load data
load logmap.mat;

% Define parameter ranges to optimize
orders = 1:5:26;         % Range of order values to test
gams = logspace(-1,5,20);  % Range of regularization parameter values
sig2s = logspace(-1,3,20); % Range of kernel bandwidth values

% Initialize variables to store performance for all parameter combinations
all_parameter_mse = zeros(length(orders), length(gams), length(sig2s));
valid_parameter_count = zeros(length(orders), length(gams), length(sig2s));

% Number of initial training points (minimum window size)
initial_train_size = 100;
% Number of validation windows
num_windows = 5;

% Total length of training data
total_length = length(Z);
% Size of each validation set
val_size = floor((total_length - initial_train_size) / num_windows);

fprintf('Starting expanding window cross-validation...\n');

% Iterate through all parameter combinations
for order_idx = 1:length(orders)
    order = orders(order_idx);
    fprintf('Testing order = %d\n', order);
    
    % Perform expanding window cross-validation
    for window = 1:num_windows
        % Calculate the end index for the current training window
        train_end = initial_train_size + (window-1) * val_size;
        
        % Define validation set
        val_start = train_end + 1;
        val_end = min(train_end + val_size, total_length);
        
        % Extract training and validation data for this window
        Z_train = Z(1:train_end);
        Z_val = Z(val_start:val_end);
        
        % Create regression problem for this window
        X_window = windowize(Z_train, 1:(order + 1));
        Y_window = X_window(:, end);
        X_window = X_window(:, 1:order);
        
        % Track best parameters for this window (for reporting only)
        window_best_mse = Inf;
        window_best_gam_idx = 0;
        window_best_sig2_idx = 0;
        
        % Evaluate all gamma and sigma combinations for this window
        for gam_idx = 1:length(gams)
            gam = gams(gam_idx);
            for sig2_idx = 1:length(sig2s)
                sig2 = sig2s(sig2_idx);
                % Train model with current parameters
                [alpha, b] = trainlssvm({X_window, Y_window, 'f', gam, sig2});
                
                % Perform prediction over validation set
                Xs = Z_train(end-order+1:end);
                prediction = predict({X_window, Y_window, 'f', gam, sig2}, Xs, length(Z_val));
                
                % Calculate MSE
                mse = mean((prediction - Z_val).^2);
                
                % Accumulate MSE for this parameter combination
                all_parameter_mse(order_idx, gam_idx, sig2_idx) = ...
                    all_parameter_mse(order_idx, gam_idx, sig2_idx) + mse;
                valid_parameter_count(order_idx, gam_idx, sig2_idx) = ...
                    valid_parameter_count(order_idx, gam_idx, sig2_idx) + 1;
                
                % Update best parameters for this window (for reporting only)
                if mse < window_best_mse
                    window_best_mse = mse;
                    window_best_gam_idx = gam_idx;
                    window_best_sig2_idx = sig2_idx;
                end
            end
        end
        
        fprintf('  Window %d: Best MSE = %.6f (gam = %.2e, sig2 = %.2e)\n', ...
                window, window_best_mse, gams(window_best_gam_idx), sig2s(window_best_sig2_idx));
    end
end

% Calculate average MSE for each parameter combination
% Only average over windows where the parameter combination was valid
avg_parameter_mse = zeros(size(all_parameter_mse));
for order_idx = 1:length(orders)
    for gam_idx = 1:length(gams)
        for sig2_idx = 1:length(sig2s)
            % Only consider combinations that worked for all windows
            if valid_parameter_count(order_idx, gam_idx, sig2_idx) == num_windows
                avg_parameter_mse(order_idx, gam_idx, sig2_idx) = ...
                    all_parameter_mse(order_idx, gam_idx, sig2_idx) / num_windows;
            else
                avg_parameter_mse(order_idx, gam_idx, sig2_idx) = Inf;
            end
        end
    end
end

% Find the parameter combination with the lowest average MSE
[min_val, min_idx] = min(avg_parameter_mse(:));
[min_order_idx, min_gam_idx, min_sig2_idx] = ind2sub(size(avg_parameter_mse), min_idx);

% Extract best parameters
best_order = orders(min_order_idx);
best_gam = gams(min_gam_idx);
best_sig2 = sig2s(min_sig2_idx);
best_mse = min_val;

% Print the optimal parameters
fprintf('\nOptimal parameters found:\n');
fprintf('  Order = %d\n', best_order);
fprintf('  Gamma = %.2e\n', best_gam);
fprintf('  Sig2  = %.2e\n', best_sig2);
fprintf('  Average MSE across all validation windows = %.6f\n', best_mse);

% Train final model using the entire training set with optimal parameters
X = windowize(Z, 1:(best_order + 1));
Y = X(:, end);
X = X(:, 1:best_order);
[alpha, b] = trainlssvm({X, Y, 'f', best_gam, best_sig2});

% Predict on test set
Xs = Z(end-best_order+1:end);
nb = length(Ztest);
prediction = predict({X, Y, 'f', best_gam, best_sig2}, Xs, nb);

% Calculate test set MSE
test_mse = mean((prediction - Ztest).^2);
fprintf('Test set MSE = %.6f\n', test_mse);

% Visualize results
figure;
hold on;
plot(Ztest, 'k');             % actual future values
plot(prediction, 'r');        % predicted values
legend('Actual', 'Prediction');
title(sprintf('LS-SVM RBF-kernel (order=%d, gam=%.1f, sig2=%.1f)', best_order, best_gam, best_sig2));
subtitle(sprintf('MSE expanding cross-validation=%.4f, MSE test=%.4f', best_mse, test_mse));
hold off;

% Santa Fe Laser Dataset Analysis with Expanding Window Cross-Validation
% Analyzing order=50 and optimizing parameters (order, gam, sig2)

% Load data (assuming it's available as SantaFe.mat)
% If not available, you'll need to adjust the loading code
load santafe.mat;

% Define parameter ranges to optimize
orders = 5:5:50;         % Range of order values to test
gams = logspace(-1,3,10);  % Range of regularization parameter values
sig2s = logspace(-1,2,10); % Range of kernel bandwidth values

% Initialize variables to store performance for all parameter combinations
all_parameter_mse = zeros(length(orders), length(gams), length(sig2s));
valid_parameter_count = zeros(length(orders), length(gams), length(sig2s));

% Number of initial training points (minimum window size)
initial_train_size = 400;
% Number of validation windows
num_windows = 5;

% Total length of training data
total_length = length(Z);
% Size of each validation set
val_size = floor((total_length - initial_train_size) / num_windows);

fprintf('Starting expanding window cross-validation...\n');

% Iterate through all parameter combinations
for order_idx = 1:length(orders)
    order = orders(order_idx);
    fprintf('Testing order = %d\n', order);
    
    % Perform expanding window cross-validation
    for window = 1:num_windows
        % Calculate the end index for the current training window
        train_end = initial_train_size + (window-1) * val_size;
        
        % Define validation set
        val_start = train_end + 1;
        val_end = min(train_end + val_size, total_length);
        
        % Extract training and validation data for this window
        Z_train = Z(1:train_end);
        Z_val = Z(val_start:val_end);
        
        % Create regression problem for this window
        X_window = windowize(Z_train, 1:(order + 1));
        Y_window = X_window(:, end);
        X_window = X_window(:, 1:order);
        
        % Track best parameters for this window (for reporting only)
        window_best_mse = Inf;
        window_best_gam_idx = 0;
        window_best_sig2_idx = 0;
        
        % Evaluate all gamma and sigma combinations for this window
        for gam_idx = 1:length(gams)
            gam = gams(gam_idx);
            for sig2_idx = 1:length(sig2s)
                sig2 = sig2s(sig2_idx);
                % Train model with current parameters
                [alpha, b] = trainlssvm({X_window, Y_window, 'f', gam, sig2});
                
                % Perform prediction over validation set
                Xs = Z_train(end-order+1:end);
                prediction = predict({X_window, Y_window, 'f', gam, sig2}, Xs, length(Z_val));
                
                % Calculate MSE
                mse = mean((prediction - Z_val).^2);
                
                % Accumulate MSE for this parameter combination
                all_parameter_mse(order_idx, gam_idx, sig2_idx) = ...
                    all_parameter_mse(order_idx, gam_idx, sig2_idx) + mse;
                valid_parameter_count(order_idx, gam_idx, sig2_idx) = ...
                    valid_parameter_count(order_idx, gam_idx, sig2_idx) + 1;
                
                % Update best parameters for this window (for reporting only)
                if mse < window_best_mse
                    window_best_mse = mse;
                    window_best_gam_idx = gam_idx;
                    window_best_sig2_idx = sig2_idx;
                end
            end
        end
        
        fprintf('  Window %d: Best MSE = %.6f (gam = %.2e, sig2 = %.2e)\n', ...
                window, window_best_mse, gams(window_best_gam_idx), sig2s(window_best_sig2_idx));
    end
end

% Calculate average MSE for each parameter combination
% Only average over windows where the parameter combination was valid
avg_parameter_mse = zeros(size(all_parameter_mse));
for order_idx = 1:length(orders)
    for gam_idx = 1:length(gams)
        for sig2_idx = 1:length(sig2s)
            % Only consider combinations that worked for all windows
            if valid_parameter_count(order_idx, gam_idx, sig2_idx) == num_windows
                avg_parameter_mse(order_idx, gam_idx, sig2_idx) = ...
                    all_parameter_mse(order_idx, gam_idx, sig2_idx) / num_windows;
            else
                avg_parameter_mse(order_idx, gam_idx, sig2_idx) = Inf;
            end
        end
    end
end

% Find the parameter combination with the lowest average MSE
[min_val, min_idx] = min(avg_parameter_mse(:));
[min_order_idx, min_gam_idx, min_sig2_idx] = ind2sub(size(avg_parameter_mse), min_idx);

% Extract best parameters
best_order = orders(min_order_idx);
best_gam = gams(min_gam_idx);
best_sig2 = sig2s(min_sig2_idx);
best_mse = min_val;

% Print the optimal parameters
fprintf('\nOptimal parameters found:\n');
fprintf('  Order = %d\n', best_order);
fprintf('  Gamma = %.2e\n', best_gam);
fprintf('  Sig2  = %.2e\n', best_sig2);
fprintf('  Average MSE across all validation windows = %.6f\n', best_mse);

% Train final model using the entire training set with optimal parameters
X = windowize(Z, 1:(best_order + 1));
Y = X(:, end);
X = X(:, 1:best_order);
[alpha, b] = trainlssvm({X, Y, 'f', best_gam, best_sig2});

% Predict on test set
Xs = Z(end-best_order+1:end);
nb = length(Ztest);
prediction = predict({X, Y, 'f', best_gam, best_sig2}, Xs, nb);

% Calculate test set MSE
test_mse = mean((prediction - Ztest).^2);
fprintf('Test set MSE = %.6f\n', test_mse);

% Visualize results
figure;
hold on;
plot(Ztest, 'k');             % actual future values
plot(prediction, 'r');        % predicted values
legend('Actual', 'Prediction');
title(sprintf('LS-SVM RBF-kernel (order=%d, gam=%.1f, sig2=%.1f)', best_order, best_gam, best_sig2));
subtitle(sprintf('MSE expanding cross-validation=%.4f, MSE test=%.4f', best_mse, test_mse));
hold off;

%15 order, 1 sigma, 1000 gamma;


% assume X is N×3, Y is N×1, and gam, sig2 are already defined

%1.3

sig2 = 0.4;
gam = 10;
X = 6.* rand (100 , 3) - 3;
Y = sinc ( X (: ,1) ) + 0.1.* randn (100 ,1) ;
[ selected , ranking ] = bay_lssvmARD ({ X , Y , 'f', gam , sig2 }) ;

% Horizontal bar plot
figure;
barh(ranking, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'YTick', 1:size(X,2), 'YDir', 'reverse');  % Reverse Y-axis to show most relevant at top
xticks(1:length(ranking));                         % X-axis ticks at each integer
xlim([0 length(ranking)+0.5]);                     % Give some spacing
xlabel('Ranking');
ylabel('Feature column');
title('Input Relevance via Bayesian ARD (Top = Most Relevant)');


type = 'function estimation';
kernel = 'RBF_kernel';

%% Forward inclusion (including 1, 1–2, 1–2–3, 2–3, 1–3)
subsets_fwd = { ...
    1, ...       % feature 1
    2, ... 
    3, ... 
    [1 2], ...   % features 1–2
    [2 3], ...   % features 2–3
    [1 3], ...    % features 1–3
    [1 2 3]
};
labels_fwd = {'1','2','3','1-2','2-3','1-3','1-2-3'};
mse_fwd = zeros(1, numel(subsets_fwd));

for k = 1:numel(subsets_fwd)
    Xi = X(:, subsets_fwd{k});
    mse_fwd(k) = crossvalidate({Xi, Y, type, gam, sig2, kernel}, 10, 'mse');
end

figure;
barh(mse_fwd, 'FaceColor', [0.5 0.7 0.5]);
set(gca, ...
    'YTick', 1:numel(labels_fwd), ...
    'YTickLabel', labels_fwd, ...
    'YDir','reverse' );
xlabel('10-Fold Cross-Validation MSE');
ylabel('Features Used');
title('Forward Feature Selection via Cross-Validation');

% Backward feature elimination (including the single‐feature subset {1})
subsets_bwd = { ...
    [1 2 3], ...  % all three
    [2 3],   ...  % drop 1
    [1 3],   ...  % drop 2
    [1 2],   ...  % drop 3
    [1]      ...  % drop 2 & 3
    [2]      ...  % drop 2 & 3
    [3]      ...  % drop 2 & 3
};
labels_bwd = {'1-2-3','2-3','1-3','1-2','1', '2', '3'};
mse_bwd = zeros(1, numel(subsets_bwd));

for k = 1:numel(subsets_bwd)
    Xi = X(:, subsets_bwd{k});
    mse_bwd(k) = crossvalidate({Xi, Y, type, gam, sig2, kernel}, 10, 'mse');
end

figure;
barh(mse_bwd, 'FaceColor', [0.5 0.7 0.5]);
set(gca, ...
    'YTick', 1:numel(labels_bwd), ...
    'YTickLabel', labels_bwd, ...
    'YDir','reverse' );
xlabel('10-Fold Cross-Validation MSE');
ylabel('Features Used');
title('Backward Feature Elimination via Cross-Validation');
