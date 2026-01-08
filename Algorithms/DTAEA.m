classdef DTAEA < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <dynamic>
% Dynamic Two-Archive Evolutionary Algorithm
% 版本：论文 100% 严格复现版 (Final Strict Reproduction)
% 包含修正：
% 1. 关联机制：使用垂直距离 (Algorithm 5)
% 2. 目标减少：仅变异填充 (Algorithm 2)
% 3. DA更新：先局部非支配排序再选最优 (Algorithm 6)

    methods
        function main(Algorithm, Problem)
            %% 1. 初始化 (Initialization)
            N = Problem.N;
            Population = Problem.Initialization();
            CA = Population;
            DA = Problem.Initialization();
            
            % 记录初始状态
            LastM = Problem.M;
            W = UniformPoint(N, LastM); 
            
            % 全局记录 (用于输出结果)
            global GLOBAL_HISTORY;
            GLOBAL_HISTORY = {};
            
            %% 2. 优化循环 (Optimization Loop)
            while Algorithm.NotTerminated(CA)
                
                GLOBAL_HISTORY{end+1} = CA.objs;
                
                % =========================================================
                % Step 1: 繁殖 (Reproduction - Algorithm 7)
                % =========================================================
                % 确保 W 维度匹配当前目标数
                if size(CA.objs, 2) ~= size(W, 2)
                    W = UniformPoint(N, size(CA.objs, 2));
                end
                
                % 计算 CA 空间占有率 (基于垂直距离关联)
                % Paper Eq. 4 & Algorithm 5
                Objs = CA.objs;
                Zmin = min(Objs, [], 1);
                Zmax = max(Objs, [], 1);
                % 归一化是计算准确距离的前提
                NormObjs = (Objs - Zmin) ./ (Zmax - Zmin + 1e-6);
                
                Region = GetAssociation(NormObjs, W);
                Occupied = length(unique(Region));
                I_CA = Occupied / size(W, 1); % Paper Section III-C
                
                % 构造交配池 (Algorithm 7)
                Parents = [];
                for i = 1 : N
                    P1 = CA(randi(length(CA)));
                    if rand() < I_CA
                        % 如果 CA 分布多样性好，双亲均来自 CA
                        P2 = CA(randi(length(CA))); 
                    else
                        % 否则，引入 DA 以增加多样性
                        P2 = DA(randi(length(DA)));
                    end
                    Parents = [Parents, P1, P2];
                end
                
                % 生成子代
                Offspring = OperatorGA(Problem, Parents);
                
                % =========================================================
                % Step 2: 动态环境响应 (Change Detection & Response)
                % =========================================================
                CurrentM = Problem.M; 
                
                % 检测环境变化：目标数改变 或 维度不一致
                if size(Offspring.objs, 2) ~= size(CA.objs, 2) || CurrentM ~= LastM
                    
                    % 1. 更新权重向量到新维度
                    W = UniformPoint(N, CurrentM);
                    
                    % 2. 重新评估存档 (获取新维度下的目标值)
                    CA = Problem.Evaluation(CA.decs);
                    DA = Problem.Evaluation(DA.decs);
                    
                    % 3. 重建策略 (Reconstruction)
                    if CurrentM > LastM
                        % === Case 1: 目标增加 (Algorithm 1) ===
                        % CA: 保持原样 (Paper: "use all optimal solutions in the last CA")
                        % DA: 使用 LHS 重置 (Paper: "replaced by randomly generated solutions via LHS")
                        DA = LatinHypercubeSampling(Problem, N);
                        
                    elseif CurrentM < LastM
                        % === Case 2: 目标减少 (Algorithm 2) ===
                        
                        % (1) 非支配排序
                        [FrontNo, ~] = NDSort(CA.objs, inf);
                        
                        % (2) 筛选: Rank 1 进 CA，Rank > 1 进 DA
                        NextCA = CA(FrontNo == 1);
                        MovedToDA = CA(FrontNo > 1);
                        
                        % (3) 重建 DA (由被淘汰的 CA 解 + LHS 填充)
                        NextDA = MovedToDA;
                        if length(NextDA) < N
                             NextDA = [NextDA, LatinHypercubeSampling(Problem, N - length(NextDA))];
                        end
                        % 如果超了，截断 (虽然论文没细说超了怎么办，通常截断)
                        if length(NextDA) > N
                             NextDA = NextDA(1:N);
                        end
                        
                        % (4) 填充 CA (基于密度的锦标赛选择 + 变异)
                        if length(NextCA) < N
                            Needed = N - length(NextCA);
                            
                            % 计算密度 (使用垂直距离)
                            Density = EstimateDensity(NextCA.objs, W);
                            
                            NewMutants = [];
                            for k = 1 : Needed
                                % Algorithm 3: Binary Tournament based on Density
                                % 优先选择位于低密度区域的解进行繁殖
                                p1 = randi(length(NextCA));
                                p2 = randi(length(NextCA));
                                
                                if Density(p1) < Density(p2)
                                    BestIdx = p1;
                                elseif Density(p1) > Density(p2)
                                    BestIdx = p2;
                                else
                                    if rand() < 0.5, BestIdx = p1; else, BestIdx = p2; end
                                end
                                
                                Parent = NextCA(BestIdx);
                                % 仅变异 (Algorithm 2 Line 5: "PolynomialMutation(x)")
                                % 参数 {0, 20, 1, 20} 意味着交叉率=0, 变异率=1
                                Mutant = OperatorGA(Problem, Parent, {0, 20, 1, 20});
                                NewMutants = [NewMutants, Mutant];
                            end
                            NextCA = [NextCA, NewMutants];
                        end
                        
                        if length(NextCA) > N
                             NextCA = NextCA(1:N);
                        end
                        
                        CA = NextCA;
                        DA = NextDA;
                    end
                    
                    LastM = CurrentM;
                    continue; % 环境变化后，跳过本次常规更新，进入下一代
                end
                
                % =========================================================
                % Step 3: 更新存档 (Update Algorithms 4 & 6)
                % =========================================================
                CA = UpdateCA(CA, Offspring, W, N);
                DA = UpdateDA(CA, DA, Offspring, W, N);
            end
            
            % 保存数据
            try
                desktop = fullfile(getenv('USERPROFILE'), 'Desktop');
                fileName = fullfile(desktop, 'DTAEA_Strict_Result.mat');
                Data = GLOBAL_HISTORY;
                save(fileName, 'Data');
            catch
                save('DTAEA_Strict_Result.mat', 'Data');
            end
        end
    end
end

%% ========================================================================
%% 核心辅助函数：垂直距离关联 (Algorithm 5)
%% ========================================================================
function Region = GetAssociation(NormObjs, W)
    % 输入: NormObjs (N x M) 已归一化的目标值
    %       W (NW x M) 权重向量
    % 输出: Region (N x 1) 每个解所属的权重向量索引
    % 引用: Paper Algorithm 5, line 2-4
    
    [N, ~] = size(NormObjs);
    [NW, ~] = size(W);
    Region = zeros(N, 1);
    
    for i = 1 : N
        x = NormObjs(i, :);
        minDist = inf;
        bestK = 1;
        
        for k = 1 : NW
            w = W(k, :);
            w_norm_sq = sum(w.^2);
            
            % 计算垂直距离 (Perpendicular Distance)
            % dist = || x - (w^T * x / ||w||^2) * w ||
            if w_norm_sq < 1e-10
                proj = zeros(size(w));
            else
                scalar_proj = (x * w') / w_norm_sq;
                proj = scalar_proj * w;
            end
            
            d_perp = norm(x - proj);
            
            if d_perp < minDist
                minDist = d_perp;
                bestK = k;
            end
        end
        Region(i) = bestK;
    end
end

%% 辅助函数：密度估计 (Algorithm 5 + Density Count)
function Density = EstimateDensity(Objs, W)
    % 1. 归一化 (Eq. 4)
    Zmin = min(Objs, [], 1);
    Zmax = max(Objs, [], 1);
    NormObjs = (Objs - Zmin) ./ (Zmax - Zmin + 1e-6);
    
    % 2. 关联 (Algorithm 5)
    Region = GetAssociation(NormObjs, W);
    
    % 3. 统计密度
    N_W = size(W, 1);
    Counts = histcounts(Region, 1:N_W+1);
    Density = Counts(Region);
end

%% 辅助函数：LHS 采样 (Algorithm 1 & 2)
function Pop = LatinHypercubeSampling(Problem, N)
    try
        X = lhsdesign(N, Problem.D);
    catch
        X = rand(N, Problem.D);
    end
    X = X .* (Problem.upper - Problem.lower) + Problem.lower;
    Pop = Problem.Evaluation(X);
end

%% 辅助函数：UpdateCA (Algorithm 4)
function NewCA = UpdateCA(CA, Q, W, N)
    R = [CA, Q];
    [FrontNo, ~] = NDSort(R.objs, inf);
    
    % 选取前几层 (Lines 3-5)
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
    
    % 如果超出 N，进行截断 (Lines 8-16)
    Objs = Candidates.objs;
    Zmin = min(Objs, [], 1);
    Zmax = max(Objs, [], 1);
    NormObjs = (Objs - Zmin) ./ (Zmax - Zmin + 1e-6);
    
    % 使用垂直距离关联 (Line 11)
    Region = GetAssociation(NormObjs, W);
    
    CurrentSet = Candidates;
    CurrentObj = NormObjs;
    CurrentRegion = Region;
    
    while length(CurrentSet) > N
        % 找出最拥挤的区域 (Line 12)
        [Counts, Edges] = histcounts(CurrentRegion, 1:size(W,1)+1);
        [~, CrowdedIdx] = max(Counts);
        CrowdedWIdx = Edges(CrowdedIdx);
        
        InWIdx = find(CurrentRegion == CrowdedWIdx);
        
        % 找出该区域内 Tchebycheff 距离最大的解 (Line 13)
        SubW = W(CrowdedWIdx, :);
        SubW(SubW < 1e-6) = 1e-6;
        
        % Eq. 6: g^tch
        TchDist = max(abs(CurrentObj(InWIdx, :)) ./ repmat(SubW, length(InWIdx), 1), [], 2);
        
        [~, WorstLocalIdx] = max(TchDist);
        RemoveIdx = InWIdx(WorstLocalIdx);
        
        % 移除 (Line 14)
        CurrentSet(RemoveIdx) = [];
        CurrentObj(RemoveIdx, :) = [];
        CurrentRegion(RemoveIdx) = [];
    end
    NewCA = CurrentSet;
end

%% 辅助函数：UpdateDA (Algorithm 6)
function NewDA = UpdateDA(CA, DA, Q, W, N)
    R = [DA, Q];
    % 归一化 (基于 CA 和 R 的混合极值，保持一致性)
    AllObjs = [CA.objs; R.objs];
    Zmin = min(AllObjs, [], 1);
    Zmax = max(AllObjs, [], 1);
    
    NormR = (R.objs - Zmin) ./ (Zmax - Zmin + 1e-6);
    NormCA = (CA.objs - Zmin) ./ (Zmax - Zmin + 1e-6);
    
    % 使用垂直距离关联 (Line 3)
    RegionR = GetAssociation(NormR, W);
    % CA 的关联用于检查是否填满 (Line 6)
    RegionCA = GetAssociation(NormCA, W);
    
    S = [];
    RemainingIdx = 1:length(R);
    itr = 1;
    
    while length(S) < N
        for i = 1 : size(W, 1)
            if length(S) >= N, break; end
            
            % 检查 CA 在该区域的数量 (Line 6)
            CountCA = sum(RegionCA == i);
            
            if CountCA < itr
                % 在 R 的剩余解中，找到属于该区域的解
                CurrentCandidatesIdx = RemainingIdx(RegionR(RemainingIdx) == i);
                
                if ~isempty(CurrentCandidatesIdx)
                    % === Algorithm 6 Line 8: NonDominationSelection ===
                    % 严格复现：在局部子区域内，先筛选出非支配解
                    CandObjs = NormR(CurrentCandidatesIdx, :);
                    [FrontNo, ~] = NDSort(CandObjs, 1);
                    BestCandidatesIdx = CurrentCandidatesIdx(FrontNo == 1);
                    
                    % === Algorithm 6 Line 9: Argmin g^tch ===
                    % 严格复现：在非支配解中选择 Tchebycheff 距离最小的
                    SubW = W(i, :);
                    SubW(SubW < 1e-6) = 1e-6;
                    BestCandObjs = NormR(BestCandidatesIdx, :);
                    
                    TchDist = max(abs(BestCandObjs) ./ repmat(SubW, length(BestCandidatesIdx), 1), [], 2);
                    
                    [~, MinLocalIdx] = min(TchDist);
                    TargetIdx = BestCandidatesIdx(MinLocalIdx); % 映射回全局索引
                    
                    % 加入 S (Line 10)
                    S = [S, TargetIdx];
                    
                    % 从剩余列表中移除 (Line 9: O = O \ {x^b})
                    RemainingIdx(RemainingIdx == TargetIdx) = [];
                end
            end
        end
        itr = itr + 1;
        
        % 安全跳出：防止死循环 (如果迭代次数过多仍未填满)
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
