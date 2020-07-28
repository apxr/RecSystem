% =========================== inc_SVD.m ===================================
% This script is the replica of file driver_SVD.m with the only difference
% that this runs the recommender system with incremental learning

% This uses the SVD on the starting large matrix to build the recommender
% system. Once new movies are added this runs incremental learning to
% update SVD

% Input to the file: a '.mat' matlab data containing train and test data
% with name 'train_data' and 'test_data' respectively.
% Missing values should be replaced by 0
% =========================================================================


clear;

% loading data:
load ('movielens_old');

% removing columns with all 0
cols_with_all_zeros = sum(train_data == 0) == size(train_data, 1);
train = train_data(:, ~cols_with_all_zeros);
test = test_data(:, ~cols_with_all_zeros);

rateMatrix = train;
testMatrix = test;

% replacing the unrated (0) as NaN
rateMatrix(rateMatrix == 0) = NaN;


% For incremental learning : data creation --------------------------------

increment = 100; % number of columns to add as incremtal movies
% Creating the incremental matrix
incrementMatrix = rateMatrix(:, end-increment+1:end);
% removing the incremental columns from starting matrix
rateMatrix = rateMatrix(:, 1:end-increment);

% filling missing values in incremental matrix with the column average
colavg = nanmean(incrementMatrix);
colavgM = repmat(colavg, size(incrementMatrix, 1), 1);
colavgM(isnan(colavgM))=0;

incrementMatrix(isnan(incrementMatrix)) = colavgM(isnan(incrementMatrix));

% -------------------------------------------------------------------------

% Global SVD Test:
lowRank = 18;
for l=1:size(lowRank, 2)
    
    tic;
    % This part is simulating the initial learning
    
    [U, S, V, userAvg] = SVDRecommender(rateMatrix, lowRank(l));
    % comment above line and uncomment below line for running our built SVD
    % function
    % [U, S, V, userAvg] = MyJacobiSVD(rateMatrix, lowRank(l), 1);
    trainTime = toc;
    
    tic;
    % this part simulates the arrival of new movies and perform the
    % incremental learning
    current_incrementMatrix = incrementMatrix - userAvg;
    [Uk, Sk, Vk] = IncrementalLearning(U, S, V', current_incrementMatrix, ...
        lowRank(l), size(rateMatrix, 2));
    
    UV = Uk*Sk*Vk;
    UV = UV + userAvg;
    increTime=toc;
    
    % Calculating the error on train and test
    tic;
    Pred = UV;
    trainRMSE = norm((Pred - train) .* (train > 0), 'fro') / ...
        sqrt(nnz(train > 0));
    trainMAPE = sum(nansum((abs(Pred-train)./train) .* (train > 0))) ...
        / nnz(train > 0);
    testRMSE = norm((Pred - testMatrix) .* (testMatrix > 0), 'fro') ...
        / sqrt(nnz(testMatrix > 0));
    testMAPE = sum(nansum((abs(Pred-testMatrix)./testMatrix) .* ...
        (testMatrix > 0))) / nnz(testMatrix > 0);
    testRMSE_new = norm((Pred - testMatrix) .* (testMatrix > 0) .* ...
        (testMatrix > 0), 'fro') / sqrt(nnz(testMatrix > 0));
    predTime = toc;
    
    fprintf('Low rank - %d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',...
        lowRank(l), trainRMSE, testRMSE, trainMAPE, testMAPE, ...
        trainTime, increTime, predTime);
end