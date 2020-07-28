% ======================== itemCF.m =======================================
% This script runs the item based Collaborative filtering recommender sys
% using item similarities calculated on full rating matrix
% Input to the file: a '.mat' matlab data containing train and test data
% with name 'train_data' and 'test_data' respectively.
% Missing values should be replaced by 0
% =========================================================================

clear;
load ('movielens_old');

% removing columns with all 0 (only 2 are present)
cols_with_all_zeros = sum(train_data == 0) == size(train_data, 1);
train = train_data(:, ~cols_with_all_zeros);
test = test_data(:, ~cols_with_all_zeros);

tic;
[m,n] = size(train);

% Calculating similarities between the items

SIM = zeros(n,n);
SIM(SIM == 0) = NaN;

for i = 1:n
    SIM(i,i) = 1;
    for j = 1:i
        
        x = train(:,i);
        y = train(:,j);
        
        idx = (x > 0) & (y > 0);
        x = x(idx,:);
        y = y(idx,:);
        
        SIM(i,j) = x'*y/(norm(x)*norm(y));
        SIM(j,i) = SIM(i,j);
        
    end
    
end
SIM(1,1)=1;
SIM(isnan(SIM))=-2;
toc;

for L = 16:30
    
    
    tic;
    test_points = 0;
    train_points = 0;
    
    test_sqerr = 0;
    test_err = 0;
    test_ape = 0;
    train_sqerr = 0;
    train_err = 0;
    train_ape = 0;
    for i = 1:m
        for j = 1:n
            % Test error calculation
            if test(i,j)>0
                
                idx = find(train(i,:) > 0);
                [simVec, idxVec] = maxk(SIM(j,idx), L);
                
                closeMovie = idx(idxVec);
                num = simVec * (train(i, closeMovie))';
                den = sum(abs(simVec));
                
                pred = num/den;
                
                test_sqerr = test_sqerr + (pred - test(i,j))^2;
                test_err = test_err + abs(pred-test(i, j));
                test_ape = test_ape + abs(pred-test(i, j))/test(i, j);
                
                test_points = test_points+1;
            end
            % Train error calculation
            if train(i,j)>0
                
                idx = find(train(i,:) > 0);
                [simVec, idxVec] = maxk(SIM(j,idx), L);
                
                closeMovie = idx(idxVec);
                num = simVec * (train(i, closeMovie))';
                den = sum(abs(simVec));
                
                pred = num/den;
                
                train_sqerr = train_sqerr + (pred - train(i,j))^2;
                train_err = train_err + abs(pred-train(i, j));
                train_ape = train_ape + abs(pred-train(i, j))/train(i, j);
                
                train_points = train_points+1;
            end
            
        end
        
    end
    test_rmse = sqrt(test_sqerr/test_points);
    train_rmse = sqrt(train_sqerr/train_points);
    
    test_mae = test_err/test_points;
    train_mae = train_err/train_points;
    
    test_mape = test_ape/test_points;
    train_mape = train_ape/train_points;
    predTime = toc;
    fprintf('%d\t%.4f\t%.4f\t%.4f\n', L, train_rmse, test_rmse, predTime);
    
end
