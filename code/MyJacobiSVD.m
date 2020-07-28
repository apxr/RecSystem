function [U,S,V, userAvg] = MyJacobiSVD(rateMatrix, k, err)

    % This function does the SVD decomposition

    % Input: A: matrix
    %        err: required error threshold

    % Output: U, S, V: decomposed matrices

    rowavg = nanmean(rateMatrix, 2);
    colavg = nanmean(rateMatrix);

    % filling missing with the column average
    colavgM = repmat(colavg, size(rateMatrix, 1), 1);
    colavgM(isnan(colavgM))=0;

    Rfilled = rateMatrix;
    Rfilled(isnan(Rfilled)) = colavgM(isnan(rateMatrix));

    A = Rfilled - rowavg;

    if ~exist('err','var')
        err=eps*1024;
    end
    error = 100;
    [row,col] = size(A);
    if row<col
        A = A';
    end
    [m,n] = size(A);
    V = eye(n);

    % Run until columns are sufficiently orthogonal
    while error > err
        error = 0;
        % For each pair of columns - do Jacobi Transformation
        for i = 1:n-1
            for j = i+1:n
                x = A(:,i);
                y = A(:,j);
                % Find the first element A'*A [1,1] for these two columns
                aii = x'*x;
                % Find the second element A'*A [2,2] for these two columns
                bjj = y'*y;
                % Find off diagonal element A'*A [1,2] for these two columns
                cij = x'*y;

                % Find the otthogonal angle between these two - We aim to
                % make the angle 90 degrees, terefore minimize
                % cos(theta) = a.b/mod(a)*mod(b)
                error = max(error,abs(cij)/sqrt(aii*bjj));

                % Jacobi rotation - 
                % https://en.wikipedia.org/wiki/Jacobi_rotation
                % finding J(i,j,theta)
                % [c   s]
                % [ -s c]
                beta = (bjj - aii)/(2*cij);
                t = sign(beta)/(abs(beta) + sqrt(1 + beta^2));
                c = 1/sqrt(1+t^2);
                s = c*t;

                % performing the tranformation J(i,j,theta) on A to
                % determine A * J(i,j,theta).
                % A will finallly convert to A*V
                temp = A(:,i);
                A(:,i) = c*temp - s*A(:,j);
                A(:,j) = s*temp + c*A(:,j);

                % performing the tranformation J(i,j,theta) on V to
                % determine V * J(i,j,theta).
                temp = V(:,i);
                V(:,i) = c*temp - s*V(:,j);
                V(:,j) = s*temp + c*V(:,j);

            end
        end
    end

    % Here A = AV
    % Scaling to find U and S(igma) (8.6.4 in the book)
    U = zeros(size(A,1));
    S = zeros(size(A));
    for i = 1:n
        max_norm = -1;
        max_col = -1;
        % Finding column with maximum norm to keep diagonals of S sorted.
        for j =i:n
            if(norm(A(:,j))>max_norm)
                max_norm = norm(A(:,j));
                max_col = j;
            end
        end
        perm = eye(size(A,2));
        perm(:,[i max_col]) = perm(:,[max_col i]);
        A = A*perm;
        V = V*perm;
        S(i,i) = norm(A(:,i));
        U(:,i) = A(:,i)/S(i,i);
    end

    if nargout<=1
        U = S;
    end
    if row < col
        [U,V] = deal(V,U);
        S = S';
    end

    % taking k approximations
    U = U(:,1:k);
    S = S(1:k,1:k);
    V = V(:,1:k)';
    userAvg = rowavg;
end