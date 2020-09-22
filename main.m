clc; clear; close all;
best_pc = 56; subject_count =  6; fold_count = 10;
avg_accuracies = zeros(subject_count,fold_count);
cw_accuracies = zeros(subject_count,fold_count, 8);

for subject = 1:subject_count
    [data, labels] = get_raw_data(subject);
    diff = data - mean(data);
    S = diff*diff';
    [y, ~] = eig(S);
    data_size = size(data,1); full_idx = 1:1:data_size;
    fold_length = round(data_size/fold_count);
    clear diff data mu;
    
    for j = 1:fold_count
        if j ~= fold_count
            test_idx = fold_length*(j-1)+1:1:fold_length*j;
        else
            test_idx = fold_length*(j-1)+1:1:data_size;
        end
        train_idx = full_idx(~ismember(full_idx, test_idx));
        data_pca = y(:, size(y,2)-best_pc:size(y,2));
        rand_perm = randperm(size(data_pca,1))';
        data_pca = data_pca(rand_perm,:);
        labels_j = labels(rand_perm);
        train_data = data_pca(train_idx,:); test_data = data_pca(test_idx,:);
        train_labels = labels_j(train_idx,:); test_labels = labels_j(test_idx,:);
        model = fitcdiscr(train_data,train_labels,'DiscrimType','pseudoquadratic');
        predictions = predict(model,test_data);
        cm = confusionmat(test_labels,predictions);
        for k = 1:7
            d  = diag(cm);
            cw_accuracies(subject,j,k) = d(k) / sum(cm(k,:));
        end
        avg_accuracies(subject,j) = sum(diag(cm)) / (sum(cm, 'all') + 0.0);
    end
end
final_acc = mean(avg_accuracies,2);

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