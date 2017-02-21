%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function Name   : CRC
%
% Input Arguments : trMatrix -> train matrix, each col corresponding to each image
%                   tst_vector -> test vector corresponding to test image
%                   trLabels-> labels of the training images
%
% Description     : Implements CRC
%                   
%
% Output Arguments: predicted class of the test image
%
%
% Author          : Vijay Kumar, May,2012
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%5


function [predicted_labels, sci] = CRC(tstMatrix, trMatrix, trLabels, normalize)

num_train = size(trMatrix,2);
num_test = size(tstMatrix, 2);
distinct_classes = unique(trLabels);
num_classes = length(distinct_classes);

if normalize
	trMatrix = normc(trMatrix);
	tstMatrix = normc(tstMatrix);
end

predicted_labels = zeros(num_test, 1);
trMatrix_T = trMatrix';
M = inv(trMatrix_T*trMatrix + 0.001*eye(num_train))*trMatrix_T;

%{
for testno = 1 : num_test	
	x = M*tstMatrix(:, testno);
	class_error = zeros(num_classes, 1);
	for i = 1 : num_classes
		idx = find(trLabels==distinct_classes(i));
    		coeff_c = x(idx);    		        	
            	D_c = trMatrix(:, idx);    		
    		class_error(i) = norm(tstMatrix(:,testno)-D_c*coeff_c);
	end
	[min_error,sorted_ind] = sort(class_error);
	predicted_labels(testno) = distinct_classes(sorted_ind(1));	
end
%}

X = M*tstMatrix;
class_error = zeros(num_classes, num_test);
for ic = 1:num_classes	
	idx = trLabels==distinct_classes(ic);
	coeff_c = X(idx, :);
	D_c = trMatrix(:, idx);
	recon_error = tstMatrix - D_c*coeff_c;
	class_error(ic, :) = sum(recon_error.*recon_error);	
end
[min_error, min_ids] = min(class_error);
predicted_labels = distinct_classes(min_ids);

sci = zeros(num_test,1);
for i = 1:num_test
	idx = trLabels == predicted_labels(i);
	coeff_c0 = sum(abs(X(idx,i)));
	coeff_c1 = sum(abs(X(:,i)));
	sci(i) = (num_classes*(coeff_c0/coeff_c1) - 1)/(num_classes-1);			
end
		
end
