function map = EvalMAP( score, probLabels, galLabels )

map = 0;
for i =1: size(score,2)
    [~, idx] = sort(score(i,:),'descend'); 
    query_label = probLabels(i);
    
    gt_idx = find(galLabels==query_label);
%     ap = 0;
    for j = 1:length(gt_idx)
        tmp_rank(j) = find(idx ==gt_idx(j));
        
    end
    tmp_rank = sort(tmp_rank);
    ap = sum((1:j)./(tmp_rank));
    map =  map + ap/j;
    
end
map = map / i;

end