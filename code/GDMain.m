% ======================== GDmain.m =======================================
% This script builds the recommender system based on regularized SVD
% trained through gradient descent
% Input to the file: a '.mat' matlab data containing train and test data
% with name 'train_data' and 'test_data' respectively.
% Missing values should be replaced by 0
% =========================================================================

clear;

% load data:
load ('movielens_old');

% removing columns with all 0
cols_with_all_zeros = sum(train_data == 0) == size(train_data, 1);
train = train_data(:, ~cols_with_all_zeros);
test = test_data(:, ~cols_with_all_zeros);

rateMatrix = train;
testMatrix = test;

% Iteration for different lowrank
lowRank = [3:25 35 40 50];
for l=1:size(lowRank, 2)
    tic;
    % calling gradientDescent function
    [U, V] = gradientDescent(rateMatrix, lowRank(l));
    logTime = toc;
    Pred = U*V';
    
    % Predicting and calculating error
    trainTime = toc;
    tic;
    trainRMSE = norm((Pred - rateMatrix) .* (rateMatrix > 0), 'fro') ...
        / sqrt(nnz(rateMatrix > 0));
    trainMAPE = sum(nansum((abs(Pred-train)./train) .* (train > 0))) ...
        / nnz(train > 0);
    testRMSE = norm((Pred - testMatrix) .* (testMatrix > 0), 'fro') ...
        / sqrt(nnz(testMatrix > 0));
    testMAPE = sum(nansum((abs(Pred-testMatrix)./testMatrix) .* ...
        (testMatrix > 0))) / nnz(testMatrix > 0);
    predTime = toc;
    
    fprintf('SVD-%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        lowRank(l), trainRMSE, testRMSE, trainMAPE, testMAPE, trainTime, predTime);
end