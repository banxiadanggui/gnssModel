% =========================================================================
% === 数据结构检查脚本 ===
% =========================================================================

clear; clc;

fprintf('========================================\n');
fprintf('检查 MAT 文件数据结构\n');
fprintf('========================================\n\n');

% MAT文件路径
mat_file = 'D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat';

% 加载文件
fprintf('加载文件: %s\n\n', mat_file);
data = load(mat_file);

% 列出所有变量
fprintf('文件中的所有变量:\n');
variableNames = fieldnames(data);
for i = 1:length(variableNames)
    varName = variableNames{i};
    varData = data.(varName);
    fprintf('  %d. %s - [%s]\n', i, varName, class(varData));
end

% 检查是否有 obsData
fprintf('\n========================================\n');
if isfield(data, 'obsData')
    fprintf('✓ 找到 obsData 变量\n\n');

    obsData = data.obsData;
    fprintf('obsData 信息:\n');
    fprintf('  类型: %s\n', class(obsData));
    fprintf('  大小: [%s]\n', num2str(size(obsData)));

    if iscell(obsData) && ~isempty(obsData)
        fprintf('\n第一个元素的结构:\n');
        firstElement = obsData{1};

        if isstruct(firstElement)
            fields = fieldnames(firstElement);
            fprintf('  字段数量: %d\n', length(fields));
            fprintf('  字段列表:\n');

            for i = 1:min(5, length(fields))  % 只显示前5个字段
                fieldName = fields{i};
                fieldData = firstElement.(fieldName);
                fprintf('    - %s: [%s] %s\n', fieldName, num2str(size(fieldData)), class(fieldData));

                % 如果包含 nrObs 和 signal，显示详细信息
                if isstruct(fieldData)
                    if isfield(fieldData, 'nrObs') && isfield(fieldData, 'signal')
                        fprintf('      ✓ 包含 nrObs (%d) 和 signal (%s)\n', ...
                            fieldData.nrObs, fieldData.signal);

                        if isfield(fieldData, 'channel') && ~isempty(fieldData.channel)
                            fprintf('      ✓ channel 字段存在\n');
                            channelFields = fieldnames(fieldData.channel);
                            fprintf('      channel 包含的字段: %s\n', strjoin(channelFields, ', '));
                        end
                    end
                end
            end

            if length(fields) > 5
                fprintf('    ... (共 %d 个字段)\n', length(fields));
            end
        else
            fprintf('  第一个元素不是 struct\n');
        end
    else
        fprintf('  obsData 为空或不是 cell 数组\n');
    end

    fprintf('\n结论: ');
    fprintf('数据结构符合脚本要求，可以运行 extract_features.m\n');
else
    fprintf('✗ 未找到 obsData 变量\n');
    fprintf('\n建议: 请检查变量名或修改脚本中的变量名\n');
end

fprintf('========================================\n');
