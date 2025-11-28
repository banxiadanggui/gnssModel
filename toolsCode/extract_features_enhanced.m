% =========================================================================
% === 增强版特征提取脚本 ===
% === 包含更多重要参数：SNR、改正项、DOP值等 ===
% =========================================================================

clear; clc; close all;

fprintf('========================================\n');
fprintf('特征提取 - 开始处理\n');
fprintf('========================================\n\n');

% 修改为你的MAT文件路径
mat_files = "D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat";

% 初始化存储所有数据的单元格数组
allData = {};
rowIndex = 1;

% 加载MAT文件
fprintf('正在加载 MAT 文件...\n');
load(mat_files);

% 检查必需变量
if ~exist('obsData', 'var')
    error('变量 obsData 不存在！');
end

fprintf(' obsData 已加载，历元数: %d\n', length(obsData));

% 检查 navData 是否存在
hasNavData = exist('navData', 'var');
if hasNavData
    fprintf('navData 已加载，可以提取 DOP 和卫星数量\n');
else
    fprintf(' navData 不存在，将跳过 DOP 和卫星数量\n');
end

fprintf('\n开始提取特征...\n\n');

% 遍历每个历元
for i = 1:length(obsData)
    % 获取当前历元的 obsData
    currentObs = obsData{i};

    % 尝试获取当前历元的 navData（用于 DOP 和卫星数）
    epochDOP = [NaN, NaN, NaN, NaN, NaN, NaN];  % [GDOP, PDOP, HDOP, VDOP, TDOP, ?]
    epochNrSats = [NaN, NaN];  % [总数, 系统1数量]

    if hasNavData && i <= length(navData)
        currentNav = navData{i};
        if ~isempty(currentNav) && isstruct(currentNav)
            % 提取 DOP 值
            if isfield(currentNav, 'Pos') && isstruct(currentNav.Pos)
                if isfield(currentNav.Pos, 'dop')
                    dopValues = currentNav.Pos.dop;
                    if length(dopValues) >= 5
                        epochDOP = dopValues(1:6);  % 取前6个值
                    end
                end

                % 提取卫星数量
                if isfield(currentNav.Pos, 'nrSats')
                    nrSatsValues = currentNav.Pos.nrSats;
                    if length(nrSatsValues) >= 1
                        epochNrSats(1) = nrSatsValues(1);
                        if length(nrSatsValues) >= 2
                            epochNrSats(2) = nrSatsValues(2);
                        end
                    end
                end
            end
        end
    end

    % 获取当前历元的所有信号字段
    fieldNames = fieldnames(currentObs);

    % 遍历每个信号类型（如 gpsl1, gale1b）
    for j = 1:length(fieldNames)
        fieldName = fieldNames{j};
        fieldData = currentObs.(fieldName);

        % 检查是否为有效的观测数据
        if isfield(fieldData, 'nrObs') && isfield(fieldData, 'signal')
            n = fieldData.nrObs;
            signalName = fieldData.signal;

            if mod(i, 50) == 1  % 每50个历元打印一次进度
                fprintf('历元 %d/%d, 信号: %s, 观测数: %d\n', i, length(obsData), signalName, n);
            end

            % 遍历该信号的每个观测通道
            for k = 1:n
                channelK = fieldData.channel(k);

                % ==================================================
                % 提取所有可用字段（按重要性排序）
                % ==================================================

                % 基础观测值
                carrierFreq = getFieldSafe(channelK, 'carrierFreq');
                corrP = getFieldSafe(channelK, 'corrP');
                rawP = getFieldSafe(channelK, 'rawP');
                trueRange = getFieldSafe(channelK, 'trueRange');
                rangeResid = getFieldSafe(channelK, 'rangeResid');
                doppler = getFieldSafe(channelK, 'doppler');
                dopplerResid = getFieldSafe(channelK, 'dopplerResid');


                SNR = getFieldSafe(channelK, 'SNR');


                clockCorr = getFieldSafe(channelK, 'clockCorr');
                ionoCorr = getFieldSafe(channelK, 'ionoCorr');
                tropoCorr = getFieldSafe(channelK, 'tropoCorr');


                codephase = getFieldSafe(channelK, 'codephase');

                % 时间信息
                tow = getFieldSafe(channelK, 'tow');
                transmitTime = getFieldSafe(channelK, 'transmitTime');
                week = getFieldSafe(channelK, 'week');
                receiverTow = getFieldSafe(fieldData, 'receiverTow');

                % 卫星信息
                if isfield(channelK, 'SvId') && isstruct(channelK.SvId)
                    satId = getFieldSafe(channelK.SvId, 'satId');
                else
                    satId = NaN;
                end

                %  新增：数据质量标志
                bObsOk = getFieldSafe(channelK, 'bObsOk');
                bEphOk = getFieldSafe(channelK, 'bEphOk');
                bParityOk = getFieldSafe(channelK, 'bParityOk');
                bPreambleOk = getFieldSafe(channelK, 'bPreambleOk');

                % 组装数据行
                dataRow = {
                    % 基础观测 (1-7)
                    carrierFreq;
                    doppler;
                    dopplerResid;
                    rawP;
                    corrP;
                    trueRange;
                    rangeResid;

                    %  信号质量 (8)
                    SNR;

                    %  改正项 (9-11)
                    clockCorr;
                    ionoCorr;
                    tropoCorr;

                    %  码相位 (12)
                    codephase;

                    % 时间 (13-16)
                    tow;
                    transmitTime;
                    week;
                    receiverTow;

                    % 卫星ID和信号类型 (17-18)
                    satId;
                    signalName;

                    %  质量标志 (19-22)
                    bObsOk;
                    bEphOk;
                    bParityOk;
                    bPreambleOk;

                    %  DOP 值 (23-28)
                    epochDOP(1);  % GDOP
                    epochDOP(2);  % PDOP
                    epochDOP(3);  % HDOP
                    epochDOP(4);  % VDOP
                    epochDOP(5);  % TDOP
                    epochDOP(6);  % DOP6

                    %  卫星数量 (29-30)
                    epochNrSats(1);  % 总卫星数
                    epochNrSats(2);  % 系统卫星数

                    % 历元索引 (31)
                    i;  % 历元编号
                };

                % 检查关键数值字段是否含 NaN（前17个是必需的数值字段）
                hasNaN = false;
                for m = 1:7  % 只检查最关键的前7个字段
                    val = dataRow{m};
                    if isnan(val)
                        hasNaN = true;
                        break;
                    end
                end

                % 保存数据（即使某些字段为 NaN 也保存，因为 SNR 等重要）
                if ~hasNaN
                    allData(rowIndex, :) = dataRow';
                    rowIndex = rowIndex + 1;
                end
            end
        end
    end
end

% 导出到CSV文件
csvFilePath = 'D:\skill\beidou\data\processedCSV\Data_export_enhanced.csv';

% 定义表头（31个字段）
header = {'carrierFreq', 'doppler', 'dopplerResid', 'rawP', 'corrP', 'trueRange', 'rangeResid', ...
          'SNR', ...
          'clockCorr', 'ionoCorr', 'tropoCorr', ...
          'codephase', ...
          'tow', 'transmitTime', 'week', 'receiverTow', ...
          'satId', 'signalName', ...
          'bObsOk', 'bEphOk', 'bParityOk', 'bPreambleOk', ...
          'GDOP', 'PDOP', 'HDOP', 'VDOP', 'TDOP', 'DOP6', ...
          'nrSats_total', 'nrSats_system', ...
          'epoch'};

% 写入CSV
fprintf('\n正在写入 CSV 文件...\n');
writecell(header, csvFilePath, 'WriteMode', 'overwrite');
writecell(allData, csvFilePath, 'WriteMode', 'append');

fprintf('\n========================================\n');
fprintf('✓ 数据导出完成！\n');
fprintf('========================================\n');
fprintf('总行数: %d\n', rowIndex-1);
fprintf('总字段数: %d\n', length(header));
fprintf('保存路径: %s\n', csvFilePath);
fprintf('========================================\n\n');

fprintf('导出字段说明:\n');
fprintf('  基础观测: 载波频率、多普勒、伪距等\n');
fprintf('   SNR: 信号质量指标\n');
fprintf('   改正项: 钟差、电离层、对流层\n');
fprintf('   码相位: 码相位测量\n');
fprintf('   DOP值: GDOP/PDOP/HDOP/VDOP/TDOP\n');
fprintf('   卫星数: 参与定位的卫星数量\n');
fprintf('   质量标志: 观测/星历/校验状态\n');
fprintf('\n');

% =========================================================================
% 辅助函数：安全提取字段，如果不存在则返回 NaN
% =========================================================================
function value = getFieldSafe(structData, fieldName)
    if isfield(structData, fieldName)
        value = structData.(fieldName);
        % 如果是逻辑值，转换为数值
        if islogical(value)
            value = double(value);
        end
        % 如果是空，返回 NaN
        if isempty(value)
            value = NaN;
        end
    else
        value = NaN;
    end
end
