% % ----- PCA + SVM
% clc; clear; close all; % rng(1);
% split_ratio = 0.8;
% [data, labels] = get_raw_data(1);
% % data = (data - min(data, [], 2)) ./ (max(data, [], 2) - min(data, [], 2));
% diff = data - mean(data);
% S = diff*diff';
% [y, e] = eig(S);
% e = diag(e);
% clear diff data mu;
% 
% pc_range = 1:1:900;
% accuracies = zeros(length(pc_range),1);
% for i = 1:length(pc_range)
%     data_pca = y(:, size(y,2)-pc_range(i):size(y,2));
%     rand_perm = randperm(size(data_pca,1))';
%     data_pca = data_pca(rand_perm,:);
%     labels_j = labels(rand_perm);
%     leng = size(data_pca,1);
%     split = round(leng * split_ratio);
%     train_data = data_pca(1:split,:); test_data = data_pca(split:leng,:);
%     train_labels = labels_j(1:split,:); test_labels = labels_j(split:leng,:);
%     % model = fitcecoc(train_data,train_labels);
%     model = fitcdiscr(train_data,train_labels,'DiscrimType','pseudoquadratic');
%     % model = fitcdiscr(train_data,train_labels,'DiscrimType','pseudolinear');
%     predictions = predict(model,test_data);
%     cm = confusionmat(test_labels,predictions);
%     % cwa = zeros(7,1);
%     % for i = 1:7
%     %     d  = diag(cm);
%     %     cwa(i) = d(i) / sum(cm(i,:));
%     % end
%     % confusionchart(cm);
%     accuracies(i) = sum(diag(cm)) / (sum(cm, 'all') + 0.0);
% end
% figure;
% plot(pc_range, accuracies);
% xlabel('# Principal Components');
% ylabel('Accuracy (%)');
% title('Classification Accuracy with N Selected Principal Components');

% NNMF
clc; clear; close all;
[data_subject,label_subject] = get_raw_data(1);
% data_subject = (data_subject - min(data_subject, [], 2)) ./ (max(data_subject, [], 2) - min(data_subject, [], 2));
best_accuracy = 0; best_reduced_dim = 1; split_ratio = 0.8; max_dim = 20;
[data_nnmf,~] = nnmf(data_subject,max_dim);
accuracies = zeros(1,max_dim);
for reduced_dim = 1:max_dim
    data_reduced = data_subject * data_nnmf(:,1:reduced_dim);
    rand_perm = randperm(size(data_reduced,1))';
    data_reduced = data_reduced(rand_perm,:);
    labels = label_subject(rand_perm);
    split = round(length(data_reduced) * split_ratio);
    train_data = data_reduced(1:split,:); test_data = data_reduced(split:length(data_reduced),:);
    train_labels = labels(1:split,:); test_labels = labels(split:length(data_reduced),:);
    % model = fitcecoc(train_data,train_labels);
    model = fitcdiscr(train_data,train_labels,'DiscrimType','pseudoquadratic');
    % model = fitcdiscr(train_data,train_labels,'DiscrimType','pseudolinear');
    predictions = predict(model,test_data);
    cm = confusionmat(test_labels,predictions);
    accuracy = sum(diag(cm)) / (sum(cm, 'all') + 0.0);
    if accuracy > best_accuracy
        best_reduced_dim = reduced_dim;
        best_accuracy = accuracy;
    end
    accuracies(reduced_dim) = accuracy;
end
disp(['NNMF: Best Accuracy: ' num2str(best_accuracy) ', best_reduced_dim: ' num2str(best_reduced_dim)]);
figure; plot(1:1:max_dim, accuracies);
xlabel('Size of Reduced Dimension');
ylabel('Accuracy (%)');
title('Classification Accuracy with K-Reduced Dimension Matrix');

% -------------------------------------------------------------------------
function [data, labels] = get_raw_data(subject)
fname = append('raw_s', num2str(subject),'.mat');

if isfile(fname)
    load(fname);
else
    [subs] = read_datafiles('../dataset/');
    all_labels = unique(subs{1}.trial_labels{1});
    data = [];
    labels = [];
    
    for trial = 1:size(subs{subject}.trial_labels,2)
        trial_data = subs{subject}.trial_nii{trial};
        trial_labels = subs{subject}.trial_labels{trial};
        
        instance_data = zeros(size(trial_labels,1), 40*64*64);
        for instance = 1:size(trial_labels,1)
            instance_label = trial_labels{instance};
            target_idx = find(ismember(all_labels, instance_label));
            net_response = double(reshape(trial_data(:,:,:,instance), [1, 40*64*64]));
            instance_data(instance, :) = net_response;
            labels = cat(1, labels, target_idx);
        end
        data = cat(1, data, instance_data);
    end
    save(fname, 'data','labels');
end
end