% =========================================================================
% === 特征提取脚本 ===
% =========================================================================

clear; clc; close all;

fprintf('Starting data preprocessing...\n');

% 修改为你的MAT文件路径
mat_files = "D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat";

% 定义标签: 0: Clean, 1: TGS/TGD, 2: UTD, 3: MCD
labels_map = [1, 2, 3, 1];

% 初始化存储所有数据的单元格数组（预分配提高效率）
allData = {};
rowIndex = 1;

% 加载MAT文件
fprintf('Loading MAT file: %s\n', mat_files);
load(mat_files);

% 检查obsData是否存在
if ~exist('obsData', 'var')
    error('变量 obsData 不存在！请检查MAT文件内容。可用变量：');
    who
end

fprintf('obsData loaded. Length: %d\n', length(obsData));

% 第一层循环：遍历obsData中的每个元素
for i = 1:length(obsData)
    % 获取当前元素的struct（1*1 struct）
    currentStruct = obsData{i};

    % 获取struct中所有字段名（用于第二层循环）
    fieldNames = fieldnames(currentStruct);

    % 第二层循环：遍历当前struct的每个字段（j为字段索引）
    for j = 1:length(fieldNames)
        fieldName = fieldNames{j};
        fieldData = currentStruct.(fieldName);  % 获取当前字段的数据

        % 检查该字段是否包含"nrObs"和"signal"（根据需求判断是否为目标字段）
        if isfield(fieldData, 'nrObs') && isfield(fieldData, 'signal')
            n = fieldData.nrObs;  % 获取该字段下的观测数量（a_j）
            signalName = fieldData.signal;  % 获取signalName

            fprintf('Processing field: %s, nrObs: %d, signal: %s\n', fieldName, n, signalName);

            % 第三层循环：遍历每个观测（k从1到n）
            for k = 1:n
                % 提取channel中第k个元素的目标字段
                channelK = fieldData.channel(k);  % 获取第k个channel数据

                % 按顺序提取所需字段（确保字段存在）
                dataRow = {
                    channelK.carrierFreq;
                    channelK.corrP;
                    channelK.trueRange;
                    channelK.rangeResid;
                    channelK.doppler;
                    channelK.dopplerResid;
                    channelK.tow;
                    channelK.transmitTime;
                    channelK.SvId.satId;
                    fieldData.receiverTow;  % 当前字段的receiverTow
                    signalName  % signalName（用单元格存储字符串）
                };

                hasNaN = false;
                for m = 1:10  % 前10个字段（第11个是字符串）
                    val = dataRow{m};
                    if any(isnan(val))  % 若为数值且含NaN
                        hasNaN = true;
                        break;  % 发现NaN则跳出检查循环
                    end
                end

                % 若不含NaN，则保存该行数据
                if ~hasNaN
                    allData(rowIndex, :) = dataRow';  % 转置为行向量存入
                    rowIndex = rowIndex + 1;
                end
            end
        end
    end
end

% 导出到CSV文件
% 定义CSV文件路径（修改为你的输出路径）
csvFilePath = 'D:\skill\beidou\data\processedCSV\Data_export.csv';

% 写入表头
header = {'carrierFreq', 'corrP', 'trueRange', 'rangeResid', 'doppler', 'dopplerResid',...
          'tow', 'transmitTime', 'satId', 'receiverTow', 'signalName'};
writecell(header, csvFilePath, 'WriteMode', 'overwrite');

% 写入数据（使用writecell处理单元格数组数据）
writecell(allData, csvFilePath, 'WriteMode', 'append');

fprintf('\n========================================\n');
fprintf('数据导出完成！\n');
fprintf('总行数: %d\n', rowIndex-1);
fprintf('保存路径: %s\n', csvFilePath);
fprintf('========================================\n');
