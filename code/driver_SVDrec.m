% =========================== driver_SVDrec.m =============================
% This script uses the SVD to do low rank approximation of Rating matrix
% and creates a recommender system.
% Input to the file: a '.mat' matlab data containing train and test data
% with name 'train_data' and 'test_data' respectively. 
% Missing values should be replaced by 0
% =========================================================================

clear;

% Use ratings data:
load ('movielens_old');

% removing columns with all 0 (only 2 are present)
cols_with_all_zeros = sum(train_data == 0) == size(train_data, 1);
train = train_data(:, ~cols_with_all_zeros);
test = test_data(:, ~cols_with_all_zeros);

rateMatrix = train;
testMatrix = test;

% replacing the unrated (0) as NaN
rateMatrix(rateMatrix == 0) = NaN;

% Fitting the recommender system and calculating error on train and test

lowRank = [3:30 35 40 45 50]; % low rank to test for
for l=1:size(lowRank, 2)
    
    tic;
    %[U, S, V, userAvg] = SVDRecommender(rateMatrix, lowRank(l));
    
    % comment above line and uncomment below line for running our built SVD
    % function
    [U, S, V, userAvg] = MyJacobiSVD(rateMatrix, lowRank(l), 1);
    UV = U*S*V;
    UV = UV + userAvg;
    trainTime = toc;
    
    % Predicting and calculating error
    tic;
    Pred = UV;
    trainRMSE = norm((Pred - train) .* (train > 0), 'fro') ...
        / sqrt(nnz(train > 0));
    trainMAPE = sum(nansum((abs(Pred-train)./train) .* (train > 0)))...
        / nnz(train > 0);
    testRMSE = norm((Pred - testMatrix) .* (testMatrix > 0), 'fro') ...
        / sqrt(nnz(testMatrix > 0));
    testMAPE = sum(nansum((abs(Pred-testMatrix)./testMatrix) .* ...
        (testMatrix > 0))) / nnz(testMatrix > 0);
    predTime = toc;
    
    % Printing the result for each iteration (low rank value)
    fprintf('SVD-%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n', ...
        lowRank(l), trainRMSE, testRMSE, trainMAPE, testMAPE, trainTime, predTime);
end