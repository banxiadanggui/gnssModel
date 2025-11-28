% 详细检查 satData 和 navData 的内容
clear; clc;

mat_file = 'D:\skill\beidou\data\processedMAT\UTD_processed_latest.mat';
load(mat_file);

fprintf('========== 检查 satData ==========\n');
if exist('satData', 'var') && iscell(satData) && ~isempty(satData)
    % 找到第一个非空元素
    for i = 1:min(10, length(satData))
        if ~isempty(satData{i}) && isstruct(satData{i})
            fprintf('\n第 %d 个历元的 satData:\n', i);
            fields = fieldnames(satData{i});

            for j = 1:length(fields)
                fieldName = fields{j};
                fieldData = satData{i}.(fieldName);
                fprintf('\n  字段: %s\n', fieldName);

                if isstruct(fieldData)
                    subFields = fieldnames(fieldData);
                    fprintf('    子字段:\n');
                    for k = 1:length(subFields)
                        sf = subFields{k};
                        fprintf('      - %s', sf);

                        % 检查是否是数组
                        if isfield(fieldData, sf)
                            data = fieldData.(sf);
                            if isnumeric(data) || islogical(data)
                                fprintf(' [%s]', class(data));
                                if numel(data) <= 10
                                    fprintf(' = %s', mat2str(data));
                                end
                            end
                        end

                        % 标注重要字段
                        if contains(lower(sf), 'elev')
                            fprintf(' ⚠️ [高度角]');
                        elseif contains(lower(sf), 'azim')
                            fprintf(' ⚠️ [方位角]');
                        end
                        fprintf('\n');
                    end
                end
            end
            break;
        end
    end
end

fprintf('\n\n========== 检查 navData ==========\n');
if exist('navData', 'var') && iscell(navData) && ~isempty(navData)
    for i = 1:min(10, length(navData))
        if ~isempty(navData{i}) && isstruct(navData{i})
            fprintf('\n第 %d 个历元的 navData:\n', i);
            fields = fieldnames(navData{i});

            for j = 1:length(fields)
                fieldName = fields{j};
                fieldData = navData{i}.(fieldName);
                fprintf('  %s: [%s] size=%s\n', fieldName, class(fieldData), mat2str(size(fieldData)));

                % 如果是结构体，显示子字段
                if isstruct(fieldData)
                    subFields = fieldnames(fieldData);
                    fprintf('    子字段: %s\n', strjoin(subFields, ', '));

                    % 检查是否有 DOP 相关字段
                    for k = 1:length(subFields)
                        sf = subFields{k};
                        if contains(lower(sf), 'dop') || contains(lower(sf), 'sat')
                            fprintf('      ⚠️ %s = ', sf);
                            val = fieldData.(sf);
                            if isnumeric(val) && numel(val) <= 10
                                fprintf('%s\n', mat2str(val));
                            else
                                fprintf('[%s] size=%s\n', class(val), mat2str(size(val)));
                            end
                        end
                    end
                end
            end
            break;
        end
    end
end

fprintf('\n\n========== 检查 obsData.channel 的完整字段 ==========\n');
if exist('obsData', 'var') && iscell(obsData) && ~isempty(obsData)
    obs = obsData{1};
    if isstruct(obs)
        obsFields = fieldnames(obs);
        firstSignal = obs.(obsFields{1});

        if isstruct(firstSignal) && isfield(firstSignal, 'channel')
            channel = firstSignal.channel(1);
            channelFields = fieldnames(channel);

            fprintf('channel 所有字段 (%d个):\n', length(channelFields));
            for i = 1:length(channelFields)
                fprintf('  %d. %s\n', i, channelFields{i});
            end
        end
    end
end

fprintf('\n完成!\n');
