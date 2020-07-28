function [Uk, Sk, Vk] = IncrementalLearning(U, S, V, A2, k, n1)

    % This function updates the SVD using the new incremental columns.

    % Input:
    %       U, S, V: Original SVD matrices
    %       A2: incremental matrix. This should have same rows as that of
    %       original matrix
    %       k: low rank approximation
    %       n1: Columns in the original starting rating matrix

    % Output:
    %       Uk Sk Vk' : the updated SVD decomposition matrices

    n2 = size(A2, 2);

    F = [S, U'*A2];
    [Uf,Sf,Vf] = svd(F, 'econ');
    Uf = Uf(:,1:k);
    Sf = Sf(1:k,1:k);
    Vf = Vf(:,1:k);

    b = size(S, 2);
    Vk = [V, zeros(n1, n2); zeros(n2, b), eye(n2)]*Vf;
    Uk = U*Uf;
    Sk = Sf;
    Vk=Vk';

end