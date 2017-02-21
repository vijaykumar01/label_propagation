%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Function Name   : MSSRC
%
% Input Arguments : trMatrix -> train matrix, each col corresponding to each image
%                   tst_vector -> test vector corresponding to test image
%                   trLabels-> labels of the training images
%
% Description     : Implements the Sparse coding algorithm proposed in the below paper
%                   "Robust Face Recognition via Sparse Representation" by John Wright et all.
%
% Output Arguments: predicted class of the test image
%
%
% Author          : Vijay Kumar, May,2012
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5%%%%%%%%%%%%%%%%%%%%5


function [predicted_labels, sci] = SRC(tstMatrix, trMatrix, trLabels, tst_tracks, normalize)

predicted_labels = zeros(size(tstMatrix,2),1);
sci = zeros(size(tstMatrix,2),1);

unique_tracks = unique(tst_tracks);
num_tracks = length(unique_tracks);
tstMatrix2 = zeros(size(tstMatrix,1), num_tracks);
track_pos = {};
for i = 1:num_tracks
    idx = find(tst_tracks == unique_tracks(i));
    track_pos{unique_tracks(i)} = idx;
    tstMatrix2(:,i) = mean(tstMatrix(:,idx),2);
end
tstMatrix = tstMatrix2;

num_train = size(trMatrix,2);
num_test = size(tstMatrix, 2);
distinct_classes = unique(trLabels);
num_classes = length(distinct_classes);

param.mode=2; param.lambda2 =0;
param.lambda = 0.001; param.numThreads = 1;
predicted_labels = zeros(num_test, 1);
X = mexLasso(tstMatrix,trMatrix,param);
X = full(X);


%{
for testno = 1 : num_test
	x = X(:,testno);
	class_error = zeros(num_classes, 1);
	for i = 1 : num_classes
    		coeff = zeros(num_train, 1);
    		for j= 1:num_train
        		if double(trLabels(j))==double(distinct_classes(i))
            			coeff(j)=x(j);
       	 		end      
    		end
    	class_error(i)=norm(tstMatrix(:,testno)-trMatrix*coeff);
	end
	[min_error,sorted_ind] = sort(class_error);
	predicted_labels(testno) = distinct_classes(sorted_ind(1));	
end
%}

class_error = zeros(num_classes, num_test);
for ic = 1:num_classes
        idx = trLabels==distinct_classes(ic);
        coeff_c = X(idx, :);
        D_c = trMatrix(:, idx);
        recon_error = tstMatrix - D_c*coeff_c;
        class_error(ic, :) = sum(recon_error.*recon_error);
end
[min_error, min_ids] = min(class_error);
predicted_labels_1 = distinct_classes(min_ids);

sci_1 = zeros(num_test,1);
for i = 1:num_test
        idx = trLabels == predicted_labels(i);
        coeff_c0 = sum(abs(X(idx,i)));
        coeff_c1 = sum(abs(X(:,i)));
        sci(i) = (num_classes*(coeff_c0/coeff_c1) - 1)/(num_classes-1);
end


for i = 1:num_tracks    
    idx = track_pos{unique_tracks(i)};
    predicted_labels(idx) = predicted_labels_1(i);
    sci(idx) = sci_1(i);
end

