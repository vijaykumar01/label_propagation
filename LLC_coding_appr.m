% ========================================================================
% USAGE: [Coeff]=LLC_coding_appr(B,X,knn,lambda)
% Approximated Locality-constraint Linear Coding
%
% Inputs
%       B       -M x d codebook, M entries in a d-dim space
%       X       -N x d matrix, N data points in a d-dim space
%       knn     -number of nearest neighboring
%       lambda  -regulerization to improve condition
%
% Outputs
%       Coeff   -N x M matrix, each row is a code for corresponding X
%
% Jinjun Wang, march 19, 2010
% ========================================================================

function [predicted_labels, sci, Coeff] = LLC_coding_appr(B, X, BL, knn,  beta, num_train)

if ~exist('knn', 'var') || isempty(knn),
    knn = 5;
end

if ~exist('beta', 'var') || isempty(beta),
    beta = 1e-4;
end

nframe=size(X,1);
nbase=size(B,1);
distinct_classes = unique(BL);
num_classes = length(distinct_classes);


% find k nearest neighbors
XX = sum(X.*X, 2);
BB = sum(B.*B, 2);
D  = repmat(XX, 1, nbase)-2*X*B'+repmat(BB', nframe, 1);
IDX = zeros(nframe, knn);
for i = 1:nframe,
	d = D(i,:);
	[dummy, idx] = sort(d, 'ascend');
	if i>num_train
		IDX(i, :) = idx(2:knn+1);
	else
		IDX(i, :) = idx(1:knn);
	end
end

% llc approximation coding
II = eye(knn, knn);
Coeff = zeros(nframe, nbase);
for i=1:nframe
   idx = IDX(i,:);
   z = B(idx,:) - repmat(X(i,:), knn, 1);           % shift ith pt to origin
   C = z*z';                                        % local covariance
   C = C + II*beta*trace(C);                        % regularlization (K>D)
   w = C\ones(knn,1);
   w = w/sum(w);                                    % enforce sum(w)=1
   Coeff(i,idx) = w';
end


trLabels = BL; 
trMatrix = B';
tstMatrix = X';
Coeff = Coeff';
class_error = zeros(num_classes, nframe);
for ic = 1:num_classes
        idx = trLabels==distinct_classes(ic);
        coeff_c = Coeff(idx, :);
        D_c = trMatrix(:, idx);
        recon_error = tstMatrix - D_c*coeff_c;
        class_error(ic, :) = sum(recon_error.*recon_error);
end
[min_error, min_ids] = min(class_error);
predicted_labels = distinct_classes(min_ids);

sci = zeros(nframe,1);
for i = 1:nframe
        idx = trLabels == predicted_labels(i);
        coeff_c0 = sum(abs(Coeff(idx,i)));
        coeff_c1 = sum(abs(Coeff(:,i)));
        sci(i) = (num_classes*(coeff_c0/coeff_c1) - 1)/(num_classes-1);
end

