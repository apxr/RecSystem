function [ U, S, V, userAvg ] = SVDRecommender( rateMatrix, k )

    % This function fills the missing ratings, subtracts the user bias (row
    % means) and returns low rank approximation of SVD decomposition matrices 
    % with the row mean of original matrix (excluding the missing values)

    % Input: 
    %       rateMatrix: rating matrix (NaN for unrated cells). Rows are users
    %       k: low rank for approximation of SVD
    
    % Output:
    %       U S V' : the SVD decomposition matrices
    %       userAvg: user bias (row mean)
    
    rowavg = nanmean(rateMatrix, 2);
    colavg = nanmean(rateMatrix);
    
    % filling missing with the column average
    colavgM = repmat(colavg, size(rateMatrix, 1), 1);
    colavgM(isnan(colavgM))=0;
    
    Rfilled = rateMatrix;
    Rfilled(isnan(Rfilled)) = colavgM(isnan(rateMatrix));
    
    % Removing user bias
    Rnorm = Rfilled - rowavg;
    
    [U,S,V] = svd(Rnorm, 'econ');

    % Filtering the top k features
    U = U(:,1:k);
    S = S(1:k,1:k);
    V = V(:,1:k)';
    userAvg = rowavg;

end