classdef KTDMOEA < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <dynamic>
% Knowledge Transfer Dynamic Multi-Objective Evolutionary Algorithm
% theta --- 2 --- Parameter for PS expansion

%------------------------------- Reference --------------------------------
% G. Ruan, L. L. Minku, S. Menzel, B. Sendhoff, and X. Yao, Knowledge
% Transfer for Dynamic Multi-Objective Optimization With a Changing Number
% of Objectives, IEEE Transactions on Emerging Topics in Computational
% Intelligence, 2024.
%--------------------------------------------------------------------------

    properties
        theta = 2; % Parameter theta as described in Eq. (2)
    end

    methods
        function main(Algorithm, Problem)
            % Parameter setting
            Algorithm.theta = Algorithm.ParameterSet(2);
            
            % Initialize Population
            Population = Problem.Initialization();
            
            % Initialize Algorithm State
            lastM = Problem.M;
            
            % Optimization Loop
            % Algorithm.NotTerminated 会自动处理数据保存 (SavePoints)
            while Algorithm.NotTerminated(Population)
                
                % 1. Change Detection & Knowledge Transfer
                currentM = Problem.M;
                
                % 检测环境是否发生变化（目标数变化）
                if currentM ~= lastM
                    if currentM > lastM
                        % Case: Increasing NObj -> PS Expansion [Algorithm 1]
                        Population = Algorithm.PSExpansion(Population, Problem, lastM);
                    elseif currentM < lastM
                        % Case: Decreasing NObj -> PS Contraction [Algorithm 3]
                        Population = Algorithm.PSContraction(Population, Problem);
                    end
                    lastM = currentM;
                end
                
                % 2. Evolutionary Optimization Process
                % Calculate fitness (FrontNo and CrowdDis) for Mating Selection
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, inf);
                CrowdDis     = CrowdingDistance(Population.objs, FrontNo);
                
                % Mating Selection (Tournament)
                MatingPool = TournamentSelection(2, Problem.N, FrontNo, CrowdDis);
                Offspring  = OperatorGA(Problem, Population(MatingPool));
                
                % [CRITICAL]: Check for Dynamic Change happening DURING OperatorGA
                % 有些问题会在评估过程中改变维度，需要二次检查
                if size(Offspring(1).objs, 2) ~= size(Population(1).objs, 2)
                    newM = size(Offspring(1).objs, 2);
                    oldM = size(Population(1).objs, 2);
                    if newM > oldM
                        Population = Algorithm.PSExpansion(Population, Problem, oldM);
                    else
                        Population = Algorithm.PSContraction(Population, Problem);
                    end
                    lastM = newM;
                    % 注意：如果这里发生了变化，EnvironmentSelection 需要小心，
                    % 但通常下一轮循环会处理，或者在此处直接做简单的截断
                    [FrontNo, ~] = NDSort(Population.objs, 1);
                    CrowdDis = CrowdingDistance(Population.objs, FrontNo);
                    [~, Rank] = sort(CrowdDis, 'descend');
                    Population = Population(Rank(1:Problem.N));
                else
                    % 3. Environmental Selection (Standard NSGA-II style)
                    Population = Algorithm.EnvironmentalSelection([Population, Offspring], Problem.N);
                end
            end
        end

        %% Algorithm 1: PS Expansion Strategy
        function NewPop = PSExpansion(Algorithm, OldPop, Problem, oldM)
            % Extract Old Pareto Set (PS_t)
            [FrontNo,~] = NDSort(OldPop.objs, 1);
            PSt = OldPop(FrontNo == 1);
            if isempty(PSt), PSt = OldPop; end
            
            % Step 1: Search Expansion Directions (Using Algorithm 2)
            Dirs = Algorithm.SearchExpansionDirections(PSt, Problem, OldPop.decs);
            
            N = Problem.N;
            N_dir = size(Dirs, 1);
            TransferredDecs = [];
            
            if N_dir > 0
                % Step 2: Calculate N_base (Eq. 2)
                N_base = floor((N - oldM) / (N_dir * Algorithm.theta));
                if N_base < 1, N_base = 1; end
                
                % Evenly select base solutions from PSt using Crowding Distance
                [~, rank] = sort(CrowdingDistance(PSt.objs), 'descend');
                nSelect = min(length(PSt), N_base);
                BaseSols = PSt(rank(1:nSelect));
                
                % Step 3: Generate Solutions along Directions (Eq. 3)
                Decs = BaseSols.decs;
                NewDecs = [];
                for j = 1 : N_dir
                    D_vec = Dirs(j, :);
                    for i = 1 : size(Decs, 1)
                        x_i = Decs(i, :);
                        for k = 1 : Algorithm.theta
                            % Calculate C (Eq. 4)
                            diff_upper = (Problem.upper - x_i) ./ D_vec;
                            diff_lower = (Problem.lower - x_i) ./ D_vec;
                            candidates = [diff_upper, diff_lower];
                            candidates(candidates < 1e-6) = inf; % Filter invalid directions
                            C = min(candidates);
                            if isinf(C), C = 1.0; end 
                            
                            % Eq. 3
                            x_new = x_i + C * rand() * D_vec;
                            NewDecs = [NewDecs; x_new];
                        end
                    end
                end
                TransferredDecs = NewDecs;
            end
            
            % Step 4: Fill the rest (Evenly select from PSt)
            nRem = N - size(TransferredDecs, 1);
            if nRem > 0
                SelectPool = OldPop;
                if length(SelectPool) > nRem
                     [~, rank] = sort(CrowdingDistance(SelectPool.objs), 'descend');
                     FillDecs = SelectPool(rank(1:nRem)).decs;
                else
                     nNeed = nRem - length(SelectPool);
                     FillDecs = [SelectPool.decs; Problem.lower + rand(nNeed, Problem.D).*(Problem.upper - Problem.lower)];
                end
                TransferredDecs = [TransferredDecs; FillDecs];
            end
            
            if size(TransferredDecs, 1) > N
                TransferredDecs = TransferredDecs(1:N, :);
            end
            NewPop = Problem.Evaluation(TransferredDecs);
        end
        
        %% Algorithm 2: Search Expansion Direction (With Weight Vectors)
        function Dirs = SearchExpansionDirections(Algorithm, PSt, Problem, PSt_Decs)
            Dirs = [];
            
            % Line 1: Find Extreme Points
            [~, maxIdx] = max(PSt.objs, [], 1);
            Pe = PSt(unique(maxIdx)); 
            if isempty(Pe), return; end
            
            % Line 2: Select random extreme point x_e
            xe = Pe(randi(length(Pe)));
            xe_dec = xe.decs;
            
            % Line 3: Generate Detective Population P_var
            N = Problem.N;
            RepXe = repmat(xe, N, 1);
            P_var_Decs = Algorithm.PolynomialMutation(RepXe.decs, Problem.lower, Problem.upper);
            
            % Line 4: Evaluate P_var in NEW environment
            P_var = Problem.Evaluation(P_var_Decs);
            PSt_New = Problem.Evaluation(PSt_Decs); % Re-eval PSt in new env
            
            % Lines 5-8: Filter dominated solutions (P_non)
            P_non = [];
            for i = 1 : length(P_var)
                dominated = false;
                for j = 1 : length(PSt_New)
                    if all(PSt_New(j).objs <= P_var(i).objs) && any(PSt_New(j).objs < P_var(i).objs)
                        dominated = true;
                        break;
                    end
                end
                if ~dominated
                    P_non = [P_non, P_var(i)];
                end
            end
            if isempty(P_non), return; end
            
            % Line 9: Density Estimation using Weight Vectors
            % 1. Generate Weight Vectors (W)
            currentM = size(P_non(1).objs, 2);
            W = Algorithm.UniformPoint(N, currentM);
            
            % 2. Normalize Objectives for Association
            AllObjs = [PSt_New.objs; P_non.objs];
            Zmin = min(AllObjs, [], 1);
            Zmax = max(AllObjs, [], 1);
            
            Zmax(Zmax == Zmin) = Zmax(Zmax == Zmin) + 1e-6;
            
            % 3. Identify occupied subareas by PSt
            OccupiedRegions = false(1, size(W, 1));
            
            PSt_Norm = (PSt_New.objs - Zmin) ./ (Zmax - Zmin);
            for i = 1 : length(PSt_New)
                norm_sol = sqrt(sum(PSt_Norm(i,:).^2));
                if norm_sol == 0, continue; end
                
                cosine = (PSt_Norm(i,:) * W') ./ norm_sol;
                [~, regionIdx] = max(cosine);
                OccupiedRegions(regionIdx) = true;
            end
            
            % 4. Filter P_non based on Occupied Regions
            P_non_Norm = (P_non.objs - Zmin) ./ (Zmax - Zmin);
            KeepIdx = false(1, length(P_non));
            
            for i = 1 : length(P_non)
                norm_sol = sqrt(sum(P_non_Norm(i,:).^2));
                if norm_sol == 0
                     KeepIdx(i) = true;
                     continue;
                end
                cosine = (P_non_Norm(i,:) * W') ./ norm_sol;
                [~, regionIdx] = max(cosine);
                
                if ~OccupiedRegions(regionIdx)
                    KeepIdx(i) = true;
                end
            end
            P_non = P_non(KeepIdx);
            
            if isempty(P_non), return; end
            
            % Line 13: Form Directions
            Dirs = zeros(length(P_non), Problem.D);
            for i = 1 : length(P_non)
                vec = P_non(i).decs - xe_dec;
                nrm = norm(vec);
                if nrm > 1e-10
                    Dirs(i, :) = vec / nrm;
                else
                    Dirs(i, :) = vec;
                end
            end
            Dirs = unique(Dirs, 'rows');
        end
        
        %% Algorithm 3: PS Contraction Strategy
        function NewPop = PSContraction(Algorithm, OldPop, Problem)
            % Line 1: Re-evaluate Old Pop
            UpdatedOldPop = Problem.Evaluation(OldPop.decs);
            [FrontNo, ~] = NDSort(UpdatedOldPop.objs, 1);
            P_non = UpdatedOldPop(FrontNo == 1);
            if isempty(P_non), P_non = UpdatedOldPop; end
            
            TransferredDecs = P_non.decs;
            
            % Line 3: Find Extreme Points
            [~, maxIdx] = max(P_non.objs, [], 1);
            Pe_indices = unique(maxIdx);
            
            % Line 4-7: Spread Enhancement
            SpreadDecs = [];
            for i = 1 : length(Pe_indices)
                idx_e = Pe_indices(i);
                xe = P_non(idx_e);
                
                dists = pdist2(xe.objs, P_non.objs);
                dists(dists == 0) = inf;
                [~, idx_close] = min(dists);
                x_close = P_non(idx_close);
                
                vec = xe.decs - x_close.decs;
                nrm = norm(vec);
                if nrm > 1e-10
                    D = vec / nrm;
                    % Calculate C (Eq. 4)
                    diff_upper = (Problem.upper - xe.decs) ./ D;
                    diff_lower = (Problem.lower - xe.decs) ./ D;
                    cand = [diff_upper, diff_lower];
                    cand(cand < 1e-6) = inf;
                    C = min(cand);
                    if isinf(C), C = 0; end
                    
                    % Eq. 5
                    x_new = xe.decs + C * D;
                    SpreadDecs = [SpreadDecs; x_new];
                end
            end
            TransferredDecs = [TransferredDecs; SpreadDecs];
            
            % Line 8: Fill (Eq. 6)
            nNeed = Problem.N - size(TransferredDecs, 1);
            if nNeed > 0
                FillDecs = zeros(nNeed, Problem.D);
                for i = 1 : nNeed
                    p = randperm(length(P_non), 2);
                    xa = P_non(p(1)).decs;
                    xb = P_non(p(2)).decs;
                    % Eq. 6: x_new = xa + rand() * (xa - xb)
                    x_temp = xa + rand() * (xa - xb);
                    x_temp = max(x_temp, Problem.lower);
                    x_temp = min(x_temp, Problem.upper);
                    FillDecs(i, :) = x_temp;
                end
                TransferredDecs = [TransferredDecs; FillDecs];
            end
            
            if size(TransferredDecs, 1) > Problem.N
                 TransferredDecs = TransferredDecs(1:Problem.N, :);
            end
            NewPop = Problem.Evaluation(TransferredDecs);
        end
        
        %% Helper: Environmental Selection (Robust NSGA-II style)
        function Population = EnvironmentalSelection(Algorithm, Population, N)
             [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
             Next = FrontNo < MaxFNo;
             Last = find(FrontNo == MaxFNo);
             [~, Rank] = sort(CrowdingDistance(Population(Last).objs), 'descend');
             Next(Last(Rank(1:N-sum(Next)))) = true;
             Population = Population(Next);
        end
        
        %% Helper: Polynomial Mutation (Standard)
        function Offspring = PolynomialMutation(Algorithm, Decs, lower, upper)
            [N, D] = size(Decs);
            if size(lower, 1) ~= N
                lower = repmat(lower, N, 1);
                upper = repmat(upper, N, 1);
            end
            proM = 1/D;
            disM = 20;
            Site  = rand(N,D) < proM;
            mu    = rand(N,D);
            temp  = Site & mu<=0.5;
            Decs(temp) = Decs(temp)+(upper(temp)-lower(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                        (1-(Decs(temp)-lower(temp))./(upper(temp)-lower(temp))).^(disM+1)).^(1/(disM+1))-1);
            temp = Site & mu>0.5; 
            Decs(temp) = Decs(temp)+(upper(temp)-lower(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                        (1-(upper(temp)-Decs(temp))./(upper(temp)-lower(temp))).^(disM+1)).^(1/(disM+1)));
            Offspring = max(min(Decs, upper), lower);
        end

        %% Helper: UniformPoint (For Weight Vector Generation)
        function [W,N] = UniformPoint(Algorithm, N, M)
            H1 = 1;
            while nchoosek(H1+M,M-1) <= N
                H1 = H1 + 1;
            end
            W = nchoosek(1:H1+M-1,M-1) - repmat(0:M-2,nchoosek(H1+M-1,M-1),1) - 1;
            W = ([W,zeros(size(W,1),1)+H1]-[zeros(size(W,1),1),W])/H1;
            if H1 < M
                H2 = 0;
                while nchoosek(H1+M-1,M-1)+nchoosek(H2+M,M-1) <= N
                    H2 = H2 + 1;
                end
                if H2 > 0
                    W2 = nchoosek(1:H2+M-1,M-1) - repmat(0:M-2,nchoosek(H2+M-1,M-1),1) - 1;
                    W2 = ([W2,zeros(size(W2,1),1)+H2]-[zeros(size(W2,1),1),W2])/H2;
                    W  = [W;W2/2+1/(2*M)];
                end
            end
            W(W<1e-6) = 1e-6;
            N = size(W,1);
        end
    end
end
