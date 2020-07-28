function [ U, V ] = gradientDescent( rateMatrix, lowRank)

    % This function runs gradient descent to learn the matrix factorization
    % of rating matrix

    % Input:
    %      	rateMatrix: rating matrix with missing ratings as 0
    %       lowRank: low rank for approximation

    % Output:
    %       U, V: low rank matrix approximation
    
    % Parameters
    maxIter = 250; 
    learningRate = 0.0003;
    regularizer = 0.002; 
    
    % Random initialization:
    [n1, n2] = size(rateMatrix);
    U = rand(n1, lowRank) / lowRank;
    V = rand(n2, lowRank) / lowRank;

    error = realmax;
    tol = 1;
    % Gradient Descent loop:
    for i = 1:maxIter
        
        if error < tol
            break
        end
        
        non_zero = rateMatrix > 0;
        v1 = V + 2 * learningRate *(transpose(rateMatrix - U * ...
            transpose(V) .* non_zero) * U - regularizer * V);
        u1 = U + 2 * learningRate *((rateMatrix - U * transpose(V) ...
            .* non_zero) * V - regularizer * U);
        V = v1;
        U = u1;
        
        error = sum((rateMatrix - U * transpose(V) .* non_zero), 'all');
    end
    
    
end