% =========================================================================
% === 数据预处理脚本：从MAT文件到PyTorch训练数据 ===
% === 功能: 特征提取 -> 打标签 -> 拼接 -> 乱序 -> 划分 -> 保存CSV ===
% =========================================================================

clear; clc; close all;

fprintf('Starting data preprocessing for PyTorch models...\n');

% --- Step 1: 定义场景、文件名、标签和攻击时间点 ---
% ！！！请根据你的实际文件名和需求进行最后确认！！！
%scenario_names = {'glo'}; % TGD文件用于生成 TGS/TGD 标签
mat_files = 'trackData_BeiDouB1_Chapter5.mat';
% 定义标签: 0: Clean, 1: TGS/TGD, 2: UTD, 3: MCD
labels_map = [1, 2, 3, 1]; 
attack_points_seconds = [131, 120, 155, 136];

% 定义数据集划分比例
train_ratio = 0.7;
val_ratio = 0.15;
test_ratio = 0.15;

% --- Step 2: 初始化用于存储所有数据的Cell数组 ---
all_features_list = {};
all_labels_list = {};

% --- Step 3: 循环加载、提取特征并生成标签 ---
for s = 1:length(mat_files)
    %fprintf('\n--- Processing scenario: %s ---\n', scenario_names{s});
    load(mat_files);
    
    num_epochs_nav = length(navData);
    sample_rate_ms = settings.nav.navSolPeriod;
    
    % --- 特征提取 (6个核心特征) ---
    features_current_scenario = NaN(num_epochs_nav, 6);
    % ... (此处代码与我们之前验证的提取代码完全相同) ...
    num_channels_track = trackData.gpsl1.nrObs;
    num_epochs_ms = length(trackData.gpsl1.channel(1).CN0fromSNR);
    all_cno = zeros(num_epochs_ms, num_channels_track);
    all_corr_energy = zeros(num_epochs_ms, num_channels_track);
    for j = 1:num_channels_track
        all_cno(:, j) = trackData.gpsl1.channel(j).CN0fromSNR;
        ip = trackData.gpsl1.channel(j).I_P; qp = trackData.gpsl1.channel(j).Q_P;
        max_len = min([length(ip), length(qp), num_epochs_ms]);
        all_corr_energy(1:max_len, j) = sqrt(ip(1:max_len).^2 + qp(1:max_len).^2);
    end
    sampled_indices = (1:num_epochs_nav) * sample_rate_ms - (sample_rate_ms - 1);
    features_current_scenario(:, 1) = mean(all_cno(sampled_indices, :), 2);
    features_current_scenario(:, 2) = std(all_cno(sampled_indices, :), 0, 2);
    features_current_scenario(:, 3) = mean(all_corr_energy(sampled_indices, :), 2);
    rx_clock_bias_s = NaN(num_epochs_nav, 1);
    for i = 1:num_epochs_nav
        current_nav_pos = navData{i}.Pos;
        if isfield(current_nav_pos, 'rangeResid') && ~isempty(current_nav_pos.rangeResid)
            features_current_scenario(i, 4) = rms(current_nav_pos.rangeResid);
        end
        if isfield(current_nav_pos, 'dt')
            rx_clock_bias_s(i) = navData{i}.Pos.dt;
        end
    end
    features_current_scenario(:, 5) = rx_clock_bias_s;
    features_current_scenario(2:end, 6) = diff(rx_clock_bias_s) / (sample_rate_ms/1000);
    
    % --- 标签生成 ---
    labels_current_scenario = zeros(num_epochs_nav, 1); % 默认标签为 0 (Clean)
    attack_point_index = attack_points_seconds(s);
    % 将攻击点之后的所有样本标记为对应的攻击类型
    labels_current_scenario(attack_point_index:end) = labels_map(s);
    
    % --- 数据清理：去除任何包含NaN的行 ---
    % 检查特征矩阵中是否存在NaN行
    nan_rows = any(isnan(features_current_scenario), 2);
    features_current_scenario(nan_rows, :) = [];
    labels_current_scenario(nan_rows, :) = [];
    fprintf('Removed %d rows containing NaN values.\n', sum(nan_rows));
    
    % 将处理好的特征和标签存入cell数组
    all_features_list{end+1} = features_current_scenario;
    all_labels_list{end+1} = labels_current_scenario;
end

% --- Step 4: 拼接所有场景的数据 ---
full_feature_matrix = vertcat(all_features_list{:});
full_label_vector = vertcat(all_labels_list{:});

fprintf('\n--- Data Concatenation Summary ---\n');
fprintf('Total number of samples: %d\n', length(full_label_vector));
fprintf('Number of features: %d\n', size(full_feature_matrix, 2));
fprintf('Class distribution:\n');
fprintf(' - Class 0 (Clean):     %d samples\n', sum(full_label_vector == 0));
fprintf(' - Class 1 (TGS/TGD):   %d samples\n', sum(full_label_vector == 1));
fprintf(' - Class 2 (UTD):       %d samples\n', sum(full_label_vector == 2));
fprintf(' - Class 3 (MCD):       %d samples\n', sum(full_label_vector == 3));

% --- Step 5: 数据乱序 (Shuffle) ---
fprintf('\nShuffling the dataset...\n');
num_samples = size(full_feature_matrix, 1);
shuffled_indices = randperm(num_samples); % 生成一个随机的索引序列

shuffled_features = full_feature_matrix(shuffled_indices, :);
shuffled_labels = full_label_vector(shuffled_indices, :);

% --- Step 6: 数据集划分 ---
fprintf('Splitting the dataset into training, validation, and test sets...\n');

% 计算分割点
train_end_idx = floor(train_ratio * num_samples);
val_end_idx = train_end_idx + floor(val_ratio * num_samples);

% 分割数据
train_features = shuffled_features(1:train_end_idx, :);
train_labels = shuffled_labels(1:train_end_idx, :);

val_features = shuffled_features(train_end_idx+1:val_end_idx, :);
val_labels = shuffled_labels(train_end_idx+1:val_end_idx, :);

test_features = shuffled_features(val_end_idx+1:end, :);
test_labels = shuffled_labels(val_end_idx+1:end, :);

fprintf(' - Training set size:   %d samples\n', size(train_features, 1));
fprintf(' - Validation set size: %d samples\n', size(val_features, 1));
fprintf(' - Test set size:       %d samples\n', size(test_features, 1));

% --- Step 7: 保存为CSV文件 ---
fprintf('\nSaving datasets to CSV files...\n');

% 将特征和标签合并为一个表格，方便Python读取
train_table = array2table([train_features, train_labels]);
val_table = array2table([val_features, val_labels]);
test_table = array2table([test_features, test_labels]);

% 定义列名
feature_names = {
    'Mean_CNo', 'Std_Dev_CNo', 'Mean_Corr_Energy', ...
    'Residual_RMS', 'Clock_Bias', 'Clock_Drift'
};
column_names = [feature_names, {'Label'}];

train_table.Properties.VariableNames = column_names;
val_table.Properties.VariableNames = column_names;
test_table.Properties.VariableNames = column_names;

% 写入文件
writetable(train_table, 'train_dataset.csv');
writetable(val_table, 'validation_dataset.csv');
writetable(test_table, 'test_dataset.csv');

fprintf('Successfully created train_dataset.csv, validation_dataset.csv, and test_dataset.csv.\n');
fprintf('Data preprocessing is complete. You are now ready to train your PyTorch model!\n');