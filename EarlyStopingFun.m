function stop = EarlyStopingFun(info, patience) %stop If No Improvement after N Iterations
persistent bestLoss
persistent lastImprovedEpoch

stop = false;

% Only when loss is present (after initial iterations)
if ~isempty(info.TrainingLoss)

    if isempty(bestLoss)
        bestLoss = info.TrainingLoss;
        lastImprovedEpoch = info.Epoch;
    end

    % If loss gets improved → Update
    if info.TrainingLoss < bestLoss
        bestLoss = info.TrainingLoss;
        lastImprovedEpoch = info.Epoch;
    end

    % If X epoch Loss did not improve → Stop
    if info.Epoch - lastImprovedEpoch >= patience
        stop = true;
        % disp("Stopped early due to no improvement.");
    end
end
end
