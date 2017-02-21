clear;
rng(1);
train_fea = dlmread('imdb_feas.txt');
train_labels = dlmread('imdb_labels.txt'); 
test_fea = dlmread('hannah_feas.txt');
test_labels = dlmread('hannah_labels.txt');
test_tracks = dlmread('hannah_tracks.txt');
train_fea = train_fea'; test_fea = test_fea';
unique_labels = unique(test_labels);

% map arbitary labels to 1,2,3.. order
new_train_labels = train_labels;
new_test_labels = test_labels;
for i=1:length(unique_labels)   
   new_train_labels(find(train_labels==unique_labels(i))) = i;
   new_test_labels(find(test_labels==unique_labels(i))) = i;
end

train_labels = new_train_labels;
test_labels = new_test_labels;
unique_labels = unique(test_labels);
clear new_train_labels new_test_labels;

% subset of data for quick testing
if 0
 subset_train_fea = [];
 subset_test_fea = [];
 subset_train_labels = [];
 subset_test_labels = [];
 subset_test_tracks = [];
 for i=1:length(unique_labels)
   idx = find(train_labels==unique_labels(i));
   idx2 = randsample(idx, min(length(idx),30));   
   subset_train_fea = [subset_train_fea train_fea(:,idx2)];
   subset_train_labels = [subset_train_labels;train_labels(idx2)];

   idx = find(test_labels==unique_labels(i));
   idx2 = randsample(idx, min(length(idx),30));
   subset_test_fea = [subset_test_fea test_fea(:,idx2)];
   subset_test_labels = [subset_test_labels;test_labels(idx2)];
   subset_test_tracks = [subset_test_tracks;test_tracks(idx2)];
 end
 train_fea = subset_train_fea;
 test_fea = subset_test_fea;
 train_labels = subset_train_labels;
 test_labels = subset_test_labels;
 test_tracks = subset_test_tracks;
 clear subset_train_fea subset_test_fea subset_train_labels subset_test_labels subset_test_tracks;
end

% reduce the dimension
[train_fea, test_fea]= extract_features_PCA(train_fea, test_fea, 100);
% normalize features
train_fea =  train_fea./( repmat(sqrt(sum(train_fea.*train_fea)), [size(train_fea,1),1]));
test_fea =  test_fea./( repmat(sqrt(sum(test_fea.*test_fea)), [size(test_fea,1),1]));

%% K nearest neighbor
[idx, knn_pred_dist] = knnsearch(train_fea', test_fea', 'k',1);
knn_pred_class = train_labels(idx);
fprintf('KNN accuracy:%f\n',100*mean(knn_pred_class == test_labels));
idx = find(ismember(test_labels,unique(train_labels)));
fprintf('Labeled actor, KNN accuracy:%f\n',100*mean(knn_pred_class(idx) == test_labels(idx)));

%% CRC
[crc_pred_class, crc_pred_dist] = CRC(test_fea, train_fea, train_labels, 0);
fprintf('CRC accuracy:%f\n',100*mean(crc_pred_class == test_labels));
idx = find(ismember(test_labels,unique(train_labels)));
fprintf('Labeled actor, CRC accuracy:%f\n',100*mean(crc_pred_class(idx) == test_labels(idx)));

%% SRC
[src_pred_class, src_pred_dist] = SRC(test_fea, train_fea, train_labels, 0);
fprintf('SRC accuracy:%f\n',100*mean(src_pred_class == test_labels));
idx = find(ismember(test_labels,unique(train_labels)));
fprintf('Labeled actor, SRC accuracy:%f\n',100*mean(src_pred_class(idx) == test_labels(idx)));
%}

%% MSSRC
[mssrc_pred_class, mssrc_pred_dist] = MSSRC(test_fea, train_fea, train_labels, test_tracks, 0);
fprintf('MSSRC accuracy:%f\n',100*mean(mssrc_pred_class == test_labels));
idx = find(ismember(test_labels,unique(train_labels)));
fprintf('Labeled actor, MSSRC accuracy:%f\n',100*mean(mssrc_pred_class(idx) == test_labels(idx)));

%% LLC
[llc_pred_class, llc_pred_dist] = LLC_coding_appr(train_fea', test_fea', train_labels, 50, 0.001, size(train_fea,2));
fprintf('LLC accuracy:%f\n',100*mean(llc_pred_class == test_labels));
idx = find(ismember(test_labels,unique(train_labels)));
fprintf('Labeled actor, LLC accuracy:%f\n',100*mean(llc_pred_class(idx) == test_labels(idx)));

%% SVM
addpath('/users/vijay.kumar/tools/liblinear-2.1/matlab');
model = train(double(train_labels), sparse(train_fea'), '-s 1 -c 1 -q');
[svm_pred_class, acc1, prob1] = predict(double(test_labels), sparse(test_fea'), model, '-q');
svm_pred_dist = max(prob1,[],2);
fprintf('SVM accuracy:%f\n',100*mean(svm_pred_class == test_labels));
idx = find(ismember(test_labels,unique(train_labels)));
fprintf('Labeled actor, SVM accuracy:%f\n',100*mean(svm_pred_class(idx) == test_labels(idx)));


%% LP
unique_tracks = unique(test_tracks);
num_tracks = length(unique_tracks);
test_fea_mssrc = zeros(size(test_fea,1), num_tracks);
track_pos = {};
for i = 1:num_tracks
    idx = find(test_tracks == unique_tracks(i));
    track_pos{unique_tracks(i)} = idx;
    test_fea_mssrc(:,i) = mean(test_fea(:,idx),2);
end
a= 0.7; p.mode = 2; p.lambda2 = 2; p.lambda = 0; p.numThreads = 1; p.pos = 1;
[lp_l, lp_dist] = lp(train_fea, test_fea_mssrc, train_labels, a, p, unique_labels);
lp_pred_class = zeros(length(test_labels),1);
lp_pred_dist = zeros(length(test_labels),1);
for i = 1:num_tracks
    idx = track_pos{unique_tracks(i)};
    lp_pred_class(idx) = lp_l(i);
    lp_pred_dist(idx) = lp_dist(i);
end

fprintf('LP accuracy:%f\n',100*mean(lp_pred_class==test_labels));
idx = find(ismember(test_labels,unique(train_labels)));
fprintf('Labeled actor, LP accuracy:%f\n',100*mean(lp_pred_class(idx) == test_labels(idx)));

% save result
all_pred_scores = [100-knn_pred_dist crc_pred_dist src_pred_dist mssrc_pred_dist llc_pred_dist svm_pred_dist lp_pred_dist];
all_result = [double(knn_pred_class == test_labels) double(crc_pred_class == test_labels) double(src_pred_class == test_labels) double(mssrc_pred_class == test_labels) double(llc_pred_class == test_labels) double(svm_pred_class == test_labels) double(lp_pred_class == test_labels)];
dlmwrite('hannah_scores.txt', [all_pred_scores all_result]');
