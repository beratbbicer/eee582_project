% ---SVM - base result
clc; clear; close all;
rng(1)
split_ratio = 0.8;

normal = 2;
[accuracy, cwa, cm] = SVM(split_ratio, normal)
confusionchart(cm);

function [accuracy, cwa, cm] = SVM(split_ratio, normal)
[train_data, test_data, train_labels, test_labels] = ...
    get_train_test_split(split_ratio, normal);
model = fitcecoc(train_data,train_labels);
predictions = predict(model,test_data);
cm = confusionmat(test_labels,predictions);
accuracy = sum(diag(cm)) / (sum(cm, 'all') + 0.0);
cwa = zeros(7,1);
for i = 1:7
    d  = diag(cm);
    cwa(i) = d(i) / sum(cm(i,:));
end
end

function [train_data, test_data, train_labels, test_labels] = ...
    get_train_test_split(split_ratio, normal)
[subs] = read_datafiles('ds000105-00001');
all_labels = unique(subs{1}.trial_labels{1});
data = [];
labels = [];

for subject = 1:1
    for trial = 1:size(subs{subject}.trial_labels,2)
    trial_data = subs{subject}.trial_nii{trial};
    trial_labels = subs{subject}.trial_labels{trial};

    instance_data = zeros(size(trial_labels,1), 40*64*64);
    for instance = 1:size(trial_labels,1)
        instance_label = trial_labels{instance};
        target_idx = find(ismember(all_labels, instance_label));

%         slices = 1:40;
%         n_slices = length(slices);
        net_response = double(reshape(trial_data(:,:,:,instance), [1, 40*64*64]));
        instance_data(instance, :) = net_response;
        labels = cat(1, labels, target_idx);

    end
    data = cat(1, data, instance_data);
    end
end
clear subs;

if normal == 1
    data = (data - min(data, [], 2)) ./ (max(data, [], 2) - min(data, [], 2));
end
if normal == 2
   data = (data - mean(data)) / std(data);
end


mu = mean(data);
diff = data - mu;
S = diff*diff';
[y, e] = eig(S);
% plot_var(e);
data_pca = y(:, end-5:end);

% [~,score,~,~,explained,~] = pca(data);
% data_pca = score(:, 1:900);

rand_perm = randperm(size(data_pca,1))';
data_pca = data_pca(rand_perm,:);
labels = labels(rand_perm);
leng = size(data_pca,1);
split = round(leng * split_ratio);
train_data = data_pca(1:split,:); test_data = data_pca(split:leng,:);
train_labels = labels(1:split,:); test_labels = labels(split:leng,:);
disp('Split Complete')
end

% function plot_var(e)
% percent_var = flip(diag(e)) ./ sum(diag(e)) * 100;
% percent_var = percent_var(1:400);
% figure; plot(1:400, percent_var, 'LineWidth', 2);
% title('Individual % contribution of each eigenvector (Decreasing)')
% xlabel('eigenvectors')
% ylabel('% variance explained')
% 
% rank_vec = zeros(1, 400);
% for i = 1:400
%     rank_vec(i) = sum(percent_var(1:i));
% end
% figure; plot(1:400, rank_vec, 'LineWidth', 2);
% title('Total % variance explained by first k eigenvectors combined')
% xlabel('first k eigenvectors combined')
% ylabel('% variance explained')
% end

% ---Ablation - standardization + SVM

% ---Ablation - standardization + BG-subtract + SVM

% ---all pre-processing + ANOVA + SVM