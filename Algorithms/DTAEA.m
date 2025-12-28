classdef DTAEA < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <dynamic>
% Dynamic Two-Archive Evolutionary Algorithm

    methods
        function main(Algorithm, Problem)
            %% 1. 初始化
            N = Problem.N;
            % 初始化种群
            Population = Problem.Initialization();
            CA = Population;
            DA = Problem.Initialization();
            
            % 记录初始状态
            LastM = Problem.M;
            W = UniformPoint(N, LastM); % 初始权重
            
            % 【新增 1】声明全局变量用于记录每一代数据
            global GLOBAL_HISTORY;
            GLOBAL_HISTORY = {};
            
            %% 2. 优化循环
            while Algorithm.NotTerminated(CA)
                
                % 【新增 2】核心修改：记录当前存档 CA 的目标值
                % DTAEA 中 CA 代表收敛性最好的解集，相当于 Population
                GLOBAL_HISTORY{end+1} = CA.objs;
                
                % =========================================================
                % 步骤 1: 繁殖 (Mating)
                % =========================================================
                % 确保权重向量维度匹配
                if size(CA.objs, 2) ~= size(W, 2)
                    W = UniformPoint(N, size(CA.objs, 2));
                end
                
                % 计算 CA 的空间占有率 I_CA
                [~, Region] = min(pdist2(CA.objs, W, 'cosine'), [], 2);
                Occupied = length(unique(Region));
                I_CA = Occupied / size(W, 1);
                
                % 构造交配池 (Parents)
                Parents = [];
                for i = 1 : N
                    % 父代1 总是来自 CA
                    P1 = CA(randi(length(CA)));
                    
                    % 父代2 根据占有率来自 CA 或 DA
                    if rand() < I_CA
                        P2 = CA(randi(length(CA)));
                    else
                        P2 = DA(randi(length(DA)));
                    end
                    % 将选中的父代加入数组
                    Parents = [Parents, P1, P2];
                end
                
                % 生成并评估子代
                Offspring = OperatorGA(Problem, Parents);
                
                % =========================================================
                % 步骤 2: 动态环境响应 (Change Response)
                % =========================================================
                CurrentM = Problem.M;
                
                % 检测：如果子代评估后 M 变了，或者 CA 的维度滞后
                if CurrentM ~= LastM || size(CA.objs, 2) ~= CurrentM
                    
                    % 1. 更新权重向量
                    W = UniformPoint(N, CurrentM);
                    
                    % 2. 重新评估存档 (Re-evaluate)
                    CA = Problem.Evaluation(CA.decs);
                    DA = Problem.Evaluation(DA.decs);
                    
                    % 3. 执行重建策略 (Reconstruction)
                    if CurrentM > LastM
                        % === Case 1: 目标增加 ===
                        DA = Problem.Initialization();
                        
                    elseif CurrentM < LastM
                        % === Case 2: 目标减少 ===
                        [FrontNo, ~] = NDSort(CA.objs, 1);
                        NonDominated = CA(FrontNo == 1);
                        NextCA = NonDominated;
                        
                        if length(NextCA) < N
                            Needed = N - length(NextCA);
                            % [关键修正] 防止索引越界
                            CrowdDis = CrowdingDistance(NextCA.objs, ones(1, length(NextCA)));
                            MatingPool = TournamentSelection(2, Needed, -CrowdDis);
                            ExtraOff = OperatorGA(Problem, NextCA(MatingPool));
                            NextCA = [NextCA, ExtraOff];
                        end
                        
                        if length(NextCA) > N
                             NextCA = NextCA(1:N);
                        end
                        
                        Dominated = CA(FrontNo > 1);
                        if isempty(Dominated)
                             NextDA = Problem.Initialization(N);
                        else
                             if length(Dominated) < N
                                 NextDA = [Dominated, Problem.Initialization(N - length(Dominated))];
                             else
                                 NextDA = Dominated(1:N);
                             end
                        end
                        CA = NextCA;
                        DA = NextDA;
                    end
                    LastM = CurrentM;
                end
                
                % =========================================================
                % 步骤 3: 更新存档 (Update)
                % =========================================================
                if size(W, 2) ~= size(Offspring.objs, 2)
                    W = UniformPoint(N, size(Offspring.objs, 2));
                end
                CA = UpdateCA(CA, Offspring, W, N);
                DA = UpdateDA(CA, DA, Offspring, W, N);
            end
            
            % 【新增 3】运行结束，强制保存数据到桌面
            try
                desktop = fullfile(getenv('USERPROFILE'), 'Desktop');
                fileName = fullfile(desktop, 'DTAEA_Desktop_Result.mat');
                Data = GLOBAL_HISTORY;
                save(fileName, 'Data');
                fprintf('=======================================================\n');
                fprintf('★ DTAEA 数据已成功保存到桌面：\n%s\n', fileName);
                fprintf('=======================================================\n');
            catch
                warning('保存到桌面失败，尝试保存到当前目录...');
                save('DTAEA_Desktop_Result.mat', 'Data');
            end
        end
    end
end

%% 辅助函数 (保持不变)
function NewCA = UpdateCA(CA, Q, W, N)
    R = [CA, Q];
    [FrontNo, ~] = NDSort(R.objs, inf);
    NextCA = [];
    i = 1;
    while length(NextCA) + length(find(FrontNo==i)) <= N
        NextCA = [NextCA, R(FrontNo==i)];
        i = i + 1;
    end
    LastFront = R(FrontNo==i);
    Candidates = [NextCA, LastFront];
    if length(Candidates) <= N
        NewCA = Candidates;
        return;
    end
    Objs = Candidates.objs;
    Zmin = min(Objs, [], 1);
    Zmax = max(Objs, [], 1);
    NormObjs = (Objs - Zmin) ./ (Zmax - Zmin + 1e-6);
    [~, Region] = min(pdist2(NormObjs, W, 'cosine'), [], 2);
    CurrentSet = Candidates;
    CurrentObj = NormObjs;
    CurrentRegion = Region;
    while length(CurrentSet) > N
        [Counts, Edges] = histcounts(CurrentRegion, 1:size(W,1)+1);
        [~, CrowdedIdx] = max(Counts);
        CrowdedWIdx = Edges(CrowdedIdx);
        InWIdx = find(CurrentRegion == CrowdedWIdx);
        SubW = W(CrowdedWIdx, :);
        SubW(SubW < 1e-6) = 1e-6;
        TchDist = max(CurrentObj(InWIdx, :) ./ repmat(SubW, length(InWIdx), 1), [], 2);
        [~, WorstLocalIdx] = max(TchDist);
        RemoveIdx = InWIdx(WorstLocalIdx);
        CurrentSet(RemoveIdx) = [];
        CurrentObj(RemoveIdx, :) = [];
        CurrentRegion(RemoveIdx) = [];
    end
    NewCA = CurrentSet;
end

function NewDA = UpdateDA(CA, DA, Q, W, N)
    R = [DA, Q];
    AllObjs = [CA.objs; R.objs];
    Zmin = min(AllObjs, [], 1);
    Zmax = max(AllObjs, [], 1);
    NormR = (R.objs - Zmin) ./ (Zmax - Zmin + 1e-6);
    NormCA = (CA.objs - Zmin) ./ (Zmax - Zmin + 1e-6);
    [~, RegionR] = min(pdist2(NormR, W, 'cosine'), [], 2);
    [~, RegionCA] = min(pdist2(NormCA, W, 'cosine'), [], 2);
    S = [];
    RemainingIdx = 1:length(R);
    itr = 1;
    while length(S) < N
        for i = 1 : size(W, 1)
            if length(S) >= N, break; end
            CountCA = sum(RegionCA == i);
            if CountCA < itr
                InRIdx = find(RegionR(RemainingIdx) == i);
                RealIdx = RemainingIdx(InRIdx);
                if ~isempty(RealIdx)
                    SubW = W(i, :);
                    SubW(SubW < 1e-6) = 1e-6;
                    SubObjs = NormR(InRIdx, :);
                    TchDist = max(SubObjs ./ repmat(SubW, length(InRIdx), 1), [], 2);
                    [~, BestLocalIdx] = min(TchDist);
                    BestGlobalIdx = RealIdx(BestLocalIdx);
                    S = [S, BestGlobalIdx];
                    RemainingIdx(RemainingIdx == BestGlobalIdx) = [];
                    RegionR(RemainingIdx == BestGlobalIdx) = -1; 
                end
            end
        end
        itr = itr + 1;
        if itr > N + 10 && length(S) < N
             Rest = setdiff(1:length(R), S);
             Need = N - length(S);
             if length(Rest) >= Need, S = [S, Rest(1:Need)];
             else, S = [S, Rest]; end
             break;
        end
    end
    NewDA = R(S);

end
