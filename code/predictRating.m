function [pred] = predictRating(simVec, idxVec, u_avg, Rred, a, j, type)
    % This function predicts the rating using collaborative filtering algo

    % Input:
    %       simVec: vector of similarity values of similar items or users
    %       idxVec: index of similar users or items
    %       u_avg: user bias (average rating done by the user)
    %       Rred: processed rating matrix
    %       a: user index
    %       j: item index
    %       type: "item" for item based collaborative filtering prediction

    % Output:
    %       pred : the predicted rating of user 'a' on item 'j'

    if type == "item"
        num = simVec * (Rred(a, idxVec) +u_avg(a, :))';
        den = sum(abs(simVec));        
    else 
        num = simVec * (Rred(idxVec, j)+u_avg(idxVec, :));
        den = sum(abs(simVec));
    end
    
    pred = num/den;

end