%clear;
rng(1);
feas = dlmread('pipa_feas2');
labels = dlmread('pipa_labels2'); 

leftover_samples = find(labels==0);
feas(leftover_samples,:)=[];
labels(leftover_samples) = [];

num_examples = length(labels);
unique_labels = unique(labels);

train_idx = [];
test_idx = [];
for i =1:length(unique_labels)
    ul = unique_labels(i);
    ul_idx = find(labels == ul);
    ul_len = length(ul_idx);
    rnd_ul_idx = ul_idx(randperm(ul_len));   
    num_train = min(10, ceil(0.5*ul_len));

    train_idx = [train_idx rnd_ul_idx(1:num_train)'];
    test_idx = [test_idx rnd_ul_idx(num_train+1:ul_len)'];
end

train_fea = feas(train_idx,:)';
test_fea = feas(test_idx, :)';
train_labels = labels(train_idx);
test_labels = labels(test_idx);

% reduce the dimension
[train_fea, test_fea]= extract_features_PCA(train_fea, test_fea, 100);
% normalize features
train_fea =  train_fea./( repmat(sqrt(sum(train_fea.*train_fea)), [size(train_fea,1),1]));
test_fea =  test_fea./( repmat(sqrt(sum(test_fea.*test_fea)), [size(test_fea,1),1]));

%% K nearest neighbor
[idx, knn_pred_dist] = knnsearch(train_fea', test_fea', 'k',1);
knn_pred_class = train_labels(idx);
fprintf('KNN accuracy:%f\n',100*mean(knn_pred_class == test_labels));

%% CRC
[crc_pred_class, crc_pred_dist] = CRC(test_fea, train_fea, train_labels, 0);
fprintf('CRC accuracy:%f\n',100*mean(crc_pred_class == test_labels));

%% SRC
[src_pred_class, src_pred_dist] = SRC(test_fea, train_fea, train_labels, 0);
fprintf('SRC accuracy:%f\n',100*mean(src_pred_class == test_labels));

%% LLC
[llc_pred_class, llc_pred_dist] = LLC_coding_appr(train_fea', test_fea', train_labels, 50, 0.001, size(train_fea,2));
fprintf('LLC accuracy:%f\n',100*mean(llc_pred_class == test_labels));

%% SVM
addpath('/users/vijay.kumar/tools/liblinear-2.1/matlab');
model = train(double(train_labels), sparse(train_fea'), '-s 1 -c 1 -q');
[svm_pred_class, acc1, prob1] = predict(double(test_labels), sparse(test_fea'), model, '-q');
svm_pred_dist = max(prob1,[],2);
fprintf('SVM accuracy:%f\n',100*mean(svm_pred_class == test_labels));

%% LP
a= 0.95; p.mode = 2; p.lambda2 = 0.1; p.lambda = 0; p.numThreads = 1; p.pos = 1;
[lp_pred_class, lp_pred_dist] = lp(train_fea, test_fea, train_labels,a,p,unique_labels);
fprintf('LP accuracy:%f\n',100*mean(lp_pred_class==test_labels));

% save result
all_pred_scores = [100-knn_pred_dist crc_pred_dist src_pred_dist llc_pred_dist svm_pred_dist lp_pred_dist];
all_result = [double(knn_pred_class == test_labels) double(crc_pred_class == test_labels) double(src_pred_class == test_labels) double(llc_pred_class == test_labels) double(svm_pred_class == test_labels) double(lp_pred_class == test_labels)];
dlmwrite('pipa_scores.txt', [all_pred_scores all_result]');
