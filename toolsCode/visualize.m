% =========================================================================
% === 特征提取与可视化脚本 ===
% =========================================================================

fprintf('Starting final, verified feature extraction...\n');

% --- 确认核心变量存在 ---
if ~exist('navData', 'var') || ~exist('trackData', 'var')
    error('Required variables (navData, trackData) not found. Please load the final .mat file.');
end

% =========================================================================
% === Part 1: 从 trackData 提取高采样率特征 (C/N0, Doppler, Energy) ===
% =========================================================================
fprintf('Part 1: Extracting high-rate features from trackData...\n');

signal = 'beib1';
num_channels_track = trackData.(signal).nrObs;
num_epochs_ms = length(trackData.(signal).channel(1).CN0fromSNR); % 使用已确认的字段名

% 创建矩阵来存储所有通道的数据
all_cno = zeros(num_epochs_ms, num_channels_track);
all_doppler = zeros(num_epochs_ms, num_channels_track);
all_corr_energy = zeros(num_epochs_ms, num_channels_track);

for j = 1:num_channels_track
    all_cno(:, j) = trackData.(signal).channel(j).CN0fromSNR;
    all_doppler(:, j) = trackData.(signal).channel(j).carrFreq;
    
    ip = trackData.(signal).channel(j).I_P;
    qp = trackData.(signal).channel(j).Q_P;
    % 确保I/Q向量长度与其它向量匹配
    max_len = min([length(ip), length(qp), num_epochs_ms]);
    all_corr_energy(1:max_len, j) = sqrt(ip(1:max_len).^2 + qp(1:max_len).^2);
end

% 对高采样率数据进行降采样，以匹配导航历元
sample_rate = settings.nav.navSolPeriod; % 通常是1000ms
num_epochs_nav = length(navData);
sampled_indices = (1:num_epochs_nav) * sample_rate - (sample_rate - 1);

% 计算降采样后的特征
mean_cno = mean(all_cno(sampled_indices, :), 2);
std_cno = std(all_cno(sampled_indices, :), 0, 2);
mean_doppler = mean(all_doppler(sampled_indices, :), 2);
std_doppler = std(all_doppler(sampled_indices, :), 0, 2);
mean_corr_energy = mean(all_corr_energy(sampled_indices, :), 2);
std_corr_energy = std(all_corr_energy(sampled_indices, :), 0, 2);

% =========================================================================
% === Part 2: 从 navData 提取导航历元级特征 (Residuals, Clock) ===
% =========================================================================
fprintf('Part 2: Extracting epoch-rate features from navData...\n');

% 初始化特征向量
pseudorange_residual_rms = NaN(num_epochs_nav, 1);
rx_clock_bias_s = NaN(num_epochs_nav, 1);
rx_clock_drift_s_s = NaN(num_epochs_nav, 1); % 估算的钟漂 (s/s)

for i = 1:num_epochs_nav
    current_nav_pos = navData{i}.Pos;
    
    % 特征 7: 伪距残差RMS
    if isfield(current_nav_pos, 'rangeResid') && ~isempty(current_nav_pos.rangeResid)
        pseudorange_residual_rms(i) = rms(current_nav_pos.rangeResid);
    end
    
    % 提取钟差 (单位: 秒)
    if isfield(current_nav_pos, 'dt')
        rx_clock_bias_s(i) = current_nav_pos.dt;
    end
end

% 特征 8: 估算钟漂 (通过对钟差进行差分)
% diff(X) 计算 [X(2)-X(1), X(3)-X(2), ...]
% 我们在前面补一个NaN，使其长度与原向量一致
rx_clock_drift_s_s(2:end) = diff(rx_clock_bias_s) / (sample_rate/1000);

fprintf('All features extracted successfully!\n\n');


% =========================================================================
% === Part 3: 可视化所有已验证的特征 ===
% =========================================================================
fprintf('Plotting all verified features...\n');

time_axis = (1:num_epochs_nav) * (sample_rate / 1000); % 时间轴 (秒)

figure('Name', 'Final Verified Features - The Ground Truth', 'NumberTitle', 'off', 'Position', [100, 100, 1200, 800]);

% 绘制trackData提取的特征
subplot(4, 2, 1); plot(time_axis, mean_cno, '.-'); title('Mean C/N0 (from trackData)'); grid on; xlabel('Time (s)');
subplot(4, 2, 2); plot(time_axis, std_cno, '.-'); title('Std Dev of C/N0 (from trackData)'); grid on; xlabel('Time (s)');
subplot(4, 2, 5); plot(time_axis, mean_doppler, '.-'); title('Mean Doppler (from trackData)'); grid on; xlabel('Time (s)');
subplot(4, 2, 6); plot(time_axis, std_doppler, '.-'); title('Std Dev of Doppler (from trackData)'); grid on; xlabel('Time (s)');

% 绘制navData提取的特征
subplot(4, 2, 3); plot(time_axis, pseudorange_residual_rms, 'r.-'); title('Pseudorange Residual RMS (from navData)'); grid on; xlabel('Time (s)');
subplot(4, 2, 4); plot(time_axis, mean_corr_energy, '.-'); title('Mean Correlation Energy (from trackData)'); grid on; xlabel('Time (s)');
subplot(4, 2, 7); plot(time_axis, rx_clock_bias_s, 'm.-'); title('Receiver Clock Bias (from navData)'); grid on; xlabel('Time (s)');
subplot(4, 2, 8); plot(time_axis, rx_clock_drift_s_s, 'm.-'); title('Estimated Clock Drift (from navData)'); grid on; xlabel('Time (s)');

sgtitle('Verification of 8 Core Features (Clean versus MCD\_Spoofing)', 'FontSize', 16, 'FontWeight', 'bold');