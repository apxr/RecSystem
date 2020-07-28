% ========================== driver_itemSVD.m =============================
% This script uses the SVD to do low rank approximation of Rating matrix
% and represents Items into low dimensional space sqrt(S)*V'
%
% Input to the file: a '.mat' matlab data containing train and test data
% with name 'train_data' and 'test_data' respectively.
% Missing values should be replaced by 0

% error for different runs are stored in table named 'result'
% =========================================================================

clear;

% Loading data
load ('movielens_old');


% removing columns with all 0
cols_with_all_zeros = sum(train_data == 0) == size(train_data, 1);
train = train_data(:, ~cols_with_all_zeros);
test = test_data(:, ~cols_with_all_zeros);

[m,n] = size(train);

rateMatrix = train;
testMatrix = test;

% replacing the unrated (0) as NaN
rateMatrix(rateMatrix == 0) = NaN;

% low rank (K) and neighborhood size (L) list
k_list = [3:25 35 40 45 50];
L_list = 1:15;
result = zeros(size(k_list, 2)*size(L_list, 2),10);

for k = 1:size(k_list, 2)
    fprintf('Running Low rank - %d\n', k_list(k));
    tic;
    [U, S, V, userAvg] = SVDRecommender(rateMatrix, k_list(k));
    Sroot = sqrt(S);
    SV = Sroot*V;
    
    Rred = U*S*V;
    
    % calculating similarities
    SIM = calcSimilarity(SV, "item");
    
    trainTime = toc;
    
    for L = 1:size(L_list, 2)
        % Train
        tic;
        train_points = 0;
        test_points = 0;
        
        train_sqerr = 0;
        test_sqerr = 0;
        train_err = 0;
        test_err = 0;
        train_ape = 0;
        test_ape = 0;
        
        for movie = 1:n
            [simVec, idxVec] = maxk(SIM(movie,:), L_list(L));
            for user = 1:m
                % Train error calculation
                if train(user, movie) > 0
                    train_points = train_points+1;
                    pred = predictRating(simVec, idxVec, userAvg, Rred, ...
                        user, movie, "item");
                    train_sqerr = train_sqerr + (pred-train(user, movie))^2;
                    train_err = train_err + abs(pred-train(user, movie));
                    train_ape = train_ape + abs(pred-train(user, movie))/...
                        train(user, movie);
                    
                end
                % Test error calculation
                if test(user, movie) > 0
                    
                    test_points = test_points+1;
                    pred = predictRating(simVec, idxVec, userAvg, Rred, ...
                        user, movie, "item");
                    test_sqerr = test_sqerr + (pred - test(user, movie))^2;
                    test_err = test_err + abs(pred-test(user, movie));
                    test_ape = test_ape + abs(pred-test(user, movie))/...
                        test(user, movie);
                end
            end
        end
        predTime = toc;
        
        train_rmse = sqrt(train_sqerr/train_points);
        test_rmse = sqrt(test_sqerr/test_points);
        
        train_mae = train_err/train_points;
        test_mae = test_err/test_points;
        
        train_mape = train_ape/train_points;
        test_mape = test_ape/test_points;
        
        idx = (k-1)*size(L_list, 2)+L;
        result(idx, :) = [k_list(k) L_list(L) train_rmse test_rmse ...
            train_mape test_mape train_mae test_mae trainTime predTime];
        
    end
    
end