function [SIM] = calcSimilarity(mat, type)

    % This function calculates similarities between either user or item

    % Input:
    %       mat: low rank SVD approximated rating matrix
    %       type: "item" for item similarities calculation

    % Output:
    %       SIM : similarity matrix

    if type == "item"
        mat = mat';
    end

    norm_r = sqrt(sum(abs(mat).^2,2));

    SIM = (mat * mat.') ./ (norm_r * norm_r.');

end