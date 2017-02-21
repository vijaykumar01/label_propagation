function [pred_labels, LDS] = lp(train_fea, test_fea, train_labels, a, p, unique_labels)

D_data = [train_fea test_fea];
num_examples = size(D_data,2);
num_c = length(unique_labels);
W = zeros(num_examples, num_examples);

%compute W
for il = 1:num_examples
      %if mod(il,100)==0
	%fprintf('  coding W:%d\n', il);
      %end
      xx = D_data(:, il);
      D = D_data;
      D(:, il) = [];
      
      % solve l2 minimization with positive constraint
      coeff = mexLasso(xx, D, p);      
      coeff = full(coeff);
 
      % if the coefficients are all zeros, set weight by NN. 
      if sum(coeff) == 0                        
	[idx, dist] = knnsearch(D',xx', 'dist', 'cosine', 'k', 1);                
        coeff(idx) = 1-dist;        
      end                  
      W(il, setdiff(1:num_examples, il)) = coeff;          	 
end

%[~,~,W] = LLC_coding_appr(D_data', D_data', [train_labels;zeros(size(test_fea,2),1)], 50, 0.1, length(train_labels));
%W = W';

%[idx, dist] = knnsearch(D_data', D_data', 'dist', 'cosine', 'k', 50);
%for ix = 1:num_examples
%        W(ix, idx(ix,:)) = 1-dist(ix,:);
%end

% make it symmetric
W = 0.5*(W + W');
D = diag(sum(W,2));
S = diag(1./diag(sqrt(D)))*W*diag(1./diag(sqrt(D)));
save('S.mat', 'S', '-v7.3');

% initial label matrix
label_mat = zeros(num_examples, num_c);
for ih = 1:num_c
    indx = find(train_labels==unique_labels(ih));
    label_mat(indx, unique_labels(ih))=1;
end

% propagate
pred_scores = (eye(size(a*S))-a*S)\label_mat;

% predict labels
pred_scores = pred_scores(length(train_labels)+1:end, :);
[pred_sc_max, pred_ids] = max(pred_scores, [], 2);
pred_labels = unique_labels(pred_ids);
st_pred_scores= sort(pred_scores,2);
LDS = st_pred_scores(:,end)./sum(st_pred_scores,2);

%num_test = size(pred_scores,1);
%sci = zeros(num_test,1);
%for i = 1:num_test        
%        coeff_c0 = sum(abs(pred_scores(i,pred_labels(i))));
%        coeff_c1 = sum(abs(pred_scores(i,:)));
%        sci(i) = (num_c*(coeff_c0/coeff_c1) - 1)/(num_c-1);
%end
%LDS = sci;

end
