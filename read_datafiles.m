%datalink: https://openneuro.org/datasets/ds000105/versions/00001
%ds_name = 'ds000105-00001';

%function [subs,rest_start_timesteps,rest_end_timesteps,trial_start_timesteps, trial_end_timesteps] = read_datafiles(ds_name, folder_name_pattern)
function [subs] = read_datafiles(ds_name, sub_till_param)
% Return file data, rest start&end, trial start&end times
% Ex: ds_name = '../dataset/', folder_name_pattern = 'sub'

save_dir = 'data_mat';
if ~exist(save_dir,'dir')  
    %searcher = sprintf('*%s*', folder_name_pattern);
    searcher = sprintf('*%s*', 'sub');
    search_out = dir(fullfile(ds_name, searcher));
    s2c = struct2cell(search_out);
    subs = read_subs(ds_name, s2c, save_dir);
else
    if nargin > 1
      sub_till = sub_till_param;
    else
      sub_till = 6;
    end
    
    subs = read_mats(save_dir, sub_till);
end
end

function subs = read_mats(save_dir, sub_till)
subs = cell(sub_till, 1);
for i = 1:sub_till
    disp(['Loading subject ' num2str(i)])
    sub_base_str = [save_dir '/sub' num2str(i) '_'];
    % load([sub_base_str 'anat.mat']);
    % load([sub_base_str 'func_nii.mat']);
    % load([sub_base_str 'func_tsv.mat']);
    % subs{i}.anat = anat; 
    % subs{i}.func_niis = func_niis;
    % subs{i}.func_tsvs = func_tsvs;
    
    load([sub_base_str 'trial_nii.mat']);
    % load([sub_base_str 'rest_nii.mat']);
    load([sub_base_str 'trial_labels.mat']);
    subs{i}.trial_nii = trial_nii; 
    % subs{i}.rest_nii = rest_nii;
    subs{i}.trial_labels = trial_labels;
end
end

function subs = read_subs(ds_name, s2c, save_dir)
% subs = cell(sub_till, 3);
subs = cell(size(s2c, 2), 1);
mkdir(save_dir);

rest_start_timesteps = [1, 16, 30, 45, 59, 73, 88, 102, 117];
rest_end_timesteps = [5, 20, 34, 48, 63, 77, 92, 106, 121];
trial_start_timesteps = [6, 21, 35, 49, 64, 78, 93, 107];
trial_end_timesteps = [15, 29, 44, 58, 72, 87, 101, 116];
trial_idx = [];
for i = 1:length(trial_start_timesteps)
    trial_idx = [trial_idx, trial_start_timesteps(i):trial_end_timesteps(i)];
end
rest_idx = [];
for i = 1:length(rest_start_timesteps)
    rest_idx = [rest_idx, rest_start_timesteps(i):rest_end_timesteps(i)];
end

%for i = 1:sub_till
for i = 1:size(s2c, 2)
    disp(['Reading & Saving Subject' num2str(i)])
    subn = s2c{1, i};
    sub_path = [ds_name '/' subn '/'];
    anat = read_anat(sub_path);
    [func_niis, func_tsvs] = read_func(sub_path);
%     subs{i, 1} = anat; subs{i, 2} = func_niis; subs{i, 3} = func_tsvs;
    subs{i}.anat = anat; subs{i}.func_niis = func_niis;
    subs{i}.func_tsvs = func_tsvs;
    
    sub_base_str = [save_dir '/sub' num2str(i) '_'];
    save([sub_base_str 'anat'], 'anat');
    save([sub_base_str 'func_nii'], 'func_niis');
    save([sub_base_str 'func_tsv'], 'func_tsvs');
    
    trial_nii = cell(1, size(func_niis, 2));
    rest_nii = cell(1, size(func_niis, 2));
    trial_labels = cell(1, size(func_niis, 2));
    
    diffs = trial_end_timesteps - trial_start_timesteps + 1;
    for j = 1:size(func_niis, 2)
        unique_labels = unique(subs{i}.func_tsvs{j}.trial_type, 'rows', 'stable');
        unique_labels = cellstr(unique_labels);
        repelem(cellstr(unique_labels), diffs);

        trial_nii{j} = subs{i}.func_niis{j}(:,:,:,trial_idx);
        rest_nii{j} = subs{i}.func_niis{j}(:,:,:,rest_idx);
        trial_labels{j} = repelem(cellstr(unique_labels), diffs);
    end
    
    subs{i}.trial_nii = trial_nii;
    subs{i}.rest_nii = rest_nii;
    subs{i}.trial_labels = trial_labels;
    
    save([sub_base_str 'trial_nii'], 'trial_nii');
    save([sub_base_str 'rest_nii'], 'rest_nii');
    save([sub_base_str 'trial_labels'], 'trial_labels');
end 
end

function anat = read_anat(sub_path)
anat_path = [sub_path 'anat/'];
anat = gunzip([anat_path '*.gz']);
anat = niftiread(anat{:});
end

function [func_imgs, func_tsvs] = read_func(sub_path)
func_path = [sub_path 'func/'];

func = gunzip([func_path '*.gz']);
n_gzFiles = size(func, 2);
func_imgs = cell(1, n_gzFiles);
for i = 1:n_gzFiles
    func_imgs{i} = niftiread(func{i});
end

tsv_files = dir(fullfile([func_path '*.tsv']));
tsv_files = struct2cell(tsv_files);
n_tsvFiles = size(tsv_files, 2);
func_tsvs = cell(1, n_tsvFiles);
for i = 1:n_tsvFiles
    func_tsvs{i} = tdfread([func_path '/' tsv_files{1, i}], '\t');
end
end