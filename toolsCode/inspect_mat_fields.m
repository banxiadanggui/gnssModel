% =========================================================================
% === MAT文件字段详细检查脚本 ===
% =========================================================================

clear; clc;

fprintf('========================================\n');
fprintf('MAT 文件字段详细检查\n');
fprintf('========================================\n\n');

mat_file = 'D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat';
load(mat_file);

fprintf('检查各主要变量的字段结构...\n\n');

%% 1. 检查 trackData
fprintf('========== 1. trackData ==========\n');
if exist('trackData', 'var')
    fprintf('✓ trackData 存在\n');

    if iscell(trackData) && ~isempty(trackData)
        td = trackData{1};
        if isstruct(td)
            fields = fieldnames(td);
            fprintf('字段列表 (%d个):\n', length(fields));
            for i = 1:length(fields)
                fprintf('  %d. %s\n', i, fields{i});
            end

            % 检查第一个字段的详细结构
            fprintf('\n详细检查第一个字段: %s\n', fields{1});
            firstField = td.(fields{1});
            if isstruct(firstField)
                subFields = fieldnames(firstField);
                fprintf('  子字段 (%d个):\n', length(subFields));
                for i = 1:length(subFields)
                    sf = subFields{i};
                    fprintf('    - %s', sf);

                    % 标注重要字段
                    if contains(lower(sf), 'cn0')
                        fprintf(' ⚠️ [重要: 载噪比]');
                    elseif contains(lower(sf), 'carrier') && contains(lower(sf), 'phase')
                        fprintf(' ⚠️ [重要: 载波相位]');
                    elseif contains(lower(sf), 'pll') || contains(lower(sf), 'dll')
                        fprintf(' ⚠️ [重要: 鉴相器]');
                    elseif contains(lower(sf), 'lock')
                        fprintf(' ⚠️ [重要: 锁定状态]');
                    elseif sf == 'I_P' || sf == 'Q_P'
                        fprintf(' ⚠️ [重要: I/Q值]');
                    end
                    fprintf('\n');
                end
            end
        end
    end
else
    fprintf('✗ trackData 不存在\n');
end

%% 2. 检查 satData
fprintf('\n========== 2. satData ==========\n');
if exist('satData', 'var')
    fprintf('✓ satData 存在\n');
    fprintf('大小: [%s]\n', num2str(size(satData)));

    if iscell(satData) && ~isempty(satData)
        % 找到第一个非空元素
        for i = 1:min(10, length(satData))
            if ~isempty(satData{i}) && isstruct(satData{i})
                fprintf('\n第 %d 个元素的字段:\n', i);
                fields = fieldnames(satData{i});
                for j = 1:length(fields)
                    sf = fields{j};
                    fprintf('  - %s', sf);

                    if contains(lower(sf), 'elev')
                        fprintf(' ⚠️ [重要: 高度角]');
                    elseif contains(lower(sf), 'azim')
                        fprintf(' ⚠️ [重要: 方位角]');
                    elseif contains(lower(sf), 'pos')
                        fprintf(' [卫星位置]');
                    end
                    fprintf('\n');
                end
                break;
            end
        end
    end
else
    fprintf('✗ satData 不存在\n');
end

%% 3. 检查 navData
fprintf('\n========== 3. navData ==========\n');
if exist('navData', 'var')
    fprintf('✓ navData 存在\n');
    fprintf('大小: [%s]\n', num2str(size(navData)));

    if iscell(navData) && ~isempty(navData)
        for i = 1:min(10, length(navData))
            if ~isempty(navData{i}) && isstruct(navData{i})
                fprintf('\n第 %d 个元素的字段:\n', i);
                fields = fieldnames(navData{i});
                for j = 1:length(fields)
                    sf = fields{j};
                    fprintf('  - %s', sf);

                    if contains(lower(sf), 'dop')
                        fprintf(' ⚠️ [重要: DOP值]');
                    elseif contains(lower(sf), 'sat') && contains(lower(sf), 'nr')
                        fprintf(' ⚠️ [重要: 卫星数量]');
                    elseif contains(lower(sf), 'used')
                        fprintf(' ⚠️ [重要: 使用的卫星]');
                    end
                    fprintf('\n');
                end
                break;
            end
        end
    end
else
    fprintf('✗ navData 不存在\n');
end

%% 4. 检查 obsData (MATLAB脚本正在使用的)
fprintf('\n========== 4. obsData (MATLAB脚本使用) ==========\n');
if exist('obsData', 'var')
    fprintf('✓ obsData 存在\n');
    fprintf('MATLAB脚本正在导出的字段:\n');
    fprintf('  ✓ carrierFreq\n');
    fprintf('  ✓ corrP\n');
    fprintf('  ✓ trueRange\n');
    fprintf('  ✓ rangeResid\n');
    fprintf('  ✓ doppler\n');
    fprintf('  ✓ dopplerResid\n');
    fprintf('  ✓ tow\n');
    fprintf('  ✓ transmitTime\n');
    fprintf('  ✓ satId\n');
    fprintf('  ✓ receiverTow\n');
    fprintf('  ✓ signalName\n');

    % 检查是否有其他未导出的重要字段
    fprintf('\n检查 obsData 中是否有其他字段...\n');
    if iscell(obsData) && ~isempty(obsData)
        obs = obsData{1};
        if isstruct(obs)
            obsFields = fieldnames(obs);
            fprintf('第一个元素的字段: %s\n', strjoin(obsFields, ', '));

            % 检查第一个信号字段
            firstSignal = obs.(obsFields{1});
            if isstruct(firstSignal) && isfield(firstSignal, 'channel')
                channelFields = fieldnames(firstSignal.channel);
                fprintf('\nchannel 包含的所有字段:\n');
                for i = 1:length(channelFields)
                    cf = channelFields{i};
                    isExported = ismember(cf, {'carrierFreq', 'corrP', 'trueRange', ...
                        'rangeResid', 'doppler', 'dopplerResid', 'tow', ...
                        'transmitTime', 'SvId'});

                    if isExported
                        fprintf('  ✓ %s (已导出)\n', cf);
                    else
                        fprintf('  ✗ %s (未导出)', cf);

                        % 标注重要字段
                        if contains(lower(cf), 'cn0')
                            fprintf(' ⚠️ [重要]');
                        elseif contains(lower(cf), 'carrier') && contains(lower(cf), 'phase')
                            fprintf(' ⚠️ [重要]');
                        elseif contains(lower(cf), 'lock')
                            fprintf(' ⚠️ [重要]');
                        elseif contains(lower(cf), 'i_') || contains(lower(cf), 'q_')
                            fprintf(' ⚠️ [重要]');
                        end
                        fprintf('\n');
                    end
                end
            end
        end
    end
else
    fprintf('✗ obsData 不存在\n');
end

%% 5. 检查 statResults (Python已导出的)
fprintf('\n========== 5. statResults (Python已导出) ==========\n');
if exist('statResults', 'var')
    fprintf('✓ statResults 存在 - Python程序已导出 (62个字段)\n');
else
    fprintf('✗ statResults 不存在\n');
end

%% 总结
fprintf('\n========================================\n');
fprintf('总结与建议\n');
fprintf('========================================\n\n');

fprintf('当前状态:\n');
fprintf('  ✓ Python程序: 已导出 statResults (定位结果和统计)\n');
fprintf('  ✓ MATLAB程序: 已导出 obsData 部分字段 (观测值)\n\n');

fprintf('重要缺失:\n');
fprintf('  ⚠️  trackData 中的信号跟踪参数 (CN0, 载波相位等)\n');
fprintf('  ⚠️  satData 中的卫星几何信息 (高度角, 方位角)\n');
fprintf('  ⚠️  navData 中的导航解算信息 (DOP, 可见星数)\n');
fprintf('  ⚠️  obsData 中的其他未导出字段\n\n');

fprintf('建议: 根据需要扩展MATLAB程序，补充导出这些重要参数\n');
fprintf('========================================\n');
