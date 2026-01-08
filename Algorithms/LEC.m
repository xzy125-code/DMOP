classdef LEC < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <dynamic>
% Learning to Expand and Contract Pareto Sets
% lambda --- 20 --- Parameter for contraction sampling (popsize/20)

%------------------------------- Reference --------------------------------
% G. Ruan, L. L. Minku, S. Menzel, B. Sendhoff, and X. Yao, "Learning to
% Expand/Contract Pareto Sets in Dynamic Multiobjective Optimization With a
% Changing Number of Objectives," IEEE Transactions on Evolutionary
% Computation, 2025.
%--------------------------------------------------------------------------

    properties
        lambda = 20;    % Contraction sampling parameter
        W;              % Weight Vectors (Decomposition)
        Neighbors;      % Neighborhood index (Decomposition)
        T = 20;         % Neighborhood size
        delta = 0.9;    % Probability of choosing parents from neighborhood
    end

    methods
        function main(Algorithm, Problem)
            % Parameter setting
            Algorithm.lambda = Algorithm.ParameterSet(20);
            
            % 1. Initialization (Decomposition Strategy Setup)
            % Generate Weight Vectors and Neighborhoods based on initial M
            % [FIX]: Sync Problem.N with the generated weights
            [Algorithm.W, Problem.N] = Algorithm.UniformPoint(Problem.N, Problem.M);
            Algorithm.T = min(Problem.N, 20); 
            Algorithm.Neighbors = pdist2(Algorithm.W, Algorithm.W, 'euclidean');
            [~, Algorithm.Neighbors] = sort(Algorithm.Neighbors, 2);
            Algorithm.Neighbors = Algorithm.Neighbors(:, 1:Algorithm.T);
            
            % Initialize Population
            Population = Problem.Initialization();
            
            % Initialize Algorithm State
            lastM = Problem.M;
            
            % Optimization Loop
            while Algorithm.NotTerminated(Population)
                currentM = Problem.M;
                
                % 2. Change Detection & Knowledge Transfer
                if currentM ~= lastM
                    % [CRITICAL FIX]: Update Problem.N when Weights change!
                    [Algorithm.W, Problem.N] = Algorithm.UniformPoint(Problem.N, currentM);
                    
                    % Re-calculate Neighbors with new size
                    Algorithm.T = min(Problem.N, 20);
                    Algorithm.Neighbors = pdist2(Algorithm.W, Algorithm.W, 'euclidean');
                    [~, Algorithm.Neighbors] = sort(Algorithm.Neighbors, 2);
                    Algorithm.Neighbors = Algorithm.Neighbors(:, 1:Algorithm.T);
                    
                    if currentM > lastM
                        % Case: Increasing NObj
                        Population = Algorithm.PSExpansion(Population, Problem, lastM);
                    elseif currentM < lastM
                        % Case: Decreasing NObj
                        Population = Algorithm.PSContraction(Population, Problem, lastM);
                    end
                    lastM = currentM;
                    
                    % Ensure Population size matches the NEW Problem.N
                    if length(Population) ~= Problem.N
                         Population = Algorithm.DensitySelection(Population, Problem.N, currentM);
                    end
                end
                
                % 3. Mating Selection (Decomposition Framework / KTDMOEA style)
                MatingPool = zeros(1, Problem.N);
                for i = 1 : Problem.N
                    % [SAFEGUARD]: Ensure i does not exceed Neighbors size
                    if i > size(Algorithm.Neighbors, 1)
                        P = randi(Problem.N);
                    elseif rand < Algorithm.delta
                        P = Algorithm.Neighbors(i, randi(Algorithm.T));
                    else
                        P = randi(Problem.N);
                    end
                    MatingPool(i) = P;
                end
                
                % Generate Offspring
                Offspring = OperatorGA(Problem, Population(MatingPool));
                
                % 4. Handle Dynamic Change during Offspring Generation
                if size(Offspring(1).objs, 2) ~= size(Population(1).objs, 2)
                    newM = size(Offspring(1).objs, 2);
                    oldM = size(Population(1).objs, 2);
                    
                    % [CRITICAL FIX]: Update Problem.N here too
                    [Algorithm.W, Problem.N] = Algorithm.UniformPoint(Problem.N, newM);
                    
                    Algorithm.T = min(Problem.N, 20);
                    Algorithm.Neighbors = pdist2(Algorithm.W, Algorithm.W, 'euclidean');
                    [~, Algorithm.Neighbors] = sort(Algorithm.Neighbors, 2);
                    Algorithm.Neighbors = Algorithm.Neighbors(:, 1:Algorithm.T);
                    
                    if newM > oldM
                        Population = Algorithm.PSExpansion(Population, Problem, oldM);
                    else
                        Population = Algorithm.PSContraction(Population, Problem, oldM);
                    end
                    lastM = newM;
                    
                    % Re-select to fit new Problem.N
                    Population = Algorithm.DensitySelection(Population, Problem.N, newM);
                else
                    % 5. DTAEA Update Mechanism with Decomposition Density
                    Population = Algorithm.DensitySelection([Population, Offspring], Problem.N, currentM);
                end
            end
        end

        %% Part 1: PS Expansion
        function NewPop = PSExpansion(Algorithm, OldPop, Problem, oldM)
            % Extract Old Pareto Set
            [FrontNo,~] = NDSort(OldPop.objs, 1);
            PSt = OldPop(FrontNo == 1);
            if isempty(PSt), PSt = OldPop; end
            
            % 1. Learn Candidate Expansion Directions
            Dirs_Cand = [];
            if length(PSt) > 2
                try
                    [Coeff, ~, ~] = pca(PSt.decs);
                    nEig = min(size(Coeff, 2), oldM - 1);
                    if nEig > 0
                        EVs = Coeff(:, 1:nEig); 
                        NullBasis = null(EVs'); 
                        N = Problem.N;
                        nNull = size(NullBasis, 2);
                        if nNull > 0
                            Coefs = 2 * lhsdesign(N, nNull) - 1; 
                            Dirs_Cand = Coefs * NullBasis';
                            len = sqrt(sum(Dirs_Cand.^2, 2));
                            Dirs_Cand = Dirs_Cand ./ (len + 1e-10);
                        end
                    end
                catch
                    Dirs_Cand = [];
                end
            end
            
            % 2. Select Most Promising Directions
            D_exp = [];
            OldP_NewEnv = Problem.Evaluation(PSt.decs); % Re-evaluate
            
            if ~isempty(Dirs_Cand)
                idx = randi(length(PSt));
                x_base = PSt(idx);
                for i = 1 : size(Dirs_Cand, 1)
                    D_vec = Dirs_Cand(i, :);
                    y_dec = Algorithm.GenerateSolution(x_base.decs, D_vec, Problem.lower, Problem.upper);
                    y = Problem.Evaluation(y_dec);
                    dominated = false;
                    for k = 1 : length(OldP_NewEnv)
                        if all(OldP_NewEnv(k).objs <= y.objs) && any(OldP_NewEnv(k).objs < y.objs)
                            dominated = true;
                            break;
                        end
                    end
                    if ~dominated, D_exp = [D_exp; D_vec]; end
                end
            end
            
            % 3. Expand PS
            if isempty(D_exp)
                 NewPop = OldP_NewEnv; return; 
            end
            
            % Even Selection using local Weight Vectors for the OLD dimension
            N = Problem.N;
            nSelect = min(length(PSt), N);
            [W_old, ~] = Algorithm.UniformPoint(nSelect, oldM); 
            Objs = PSt.objs;
            [~, RegionIdx] = max(1 - pdist2(Objs, W_old, 'cosine'), [], 2);
            
            BaseSols = [];
            usedRegions = unique(RegionIdx);
            for i = 1 : length(usedRegions)
                 regionMembers = PSt(RegionIdx == usedRegions(i));
                 randIdx = randi(length(regionMembers));
                 BaseSols = [BaseSols, regionMembers(randIdx)];
            end
            if isempty(BaseSols), BaseSols = PSt; end
            
            TransferredDecs = [];
            nDirs = size(D_exp, 1);
            counter = 0;
            while size(TransferredDecs, 1) < N
                counter = counter + 1;
                idxBase = mod(counter - 1, length(BaseSols)) + 1;
                x_i = BaseSols(idxBase).decs;
                idxDir = mod(counter - 1, nDirs) + 1;
                D_vec = D_exp(idxDir, :);
                x_new = Algorithm.GenerateSolution(x_i, D_vec, Problem.lower, Problem.upper);
                TransferredDecs = [TransferredDecs; x_new];
            end
            NewPop = Problem.Evaluation(TransferredDecs(1:min(size(TransferredDecs,1), N), :));
        end

        %% Part 2: PS Contraction
        function NewPop = PSContraction(Algorithm, OldPop, Problem, oldM)
            [FrontNo,~] = NDSort(OldPop.objs, 1);
            PSt = OldPop(FrontNo == 1);
            if isempty(PSt), PSt = OldPop; end
            OldP_NewEnv = Problem.Evaluation(PSt.decs); % Re-evaluate first
            
            % 1. Learn Candidate Contraction Directions
            C_con = [];
            if length(PSt) >= oldM
                try
                     [Coeff, ~, ~] = pca(PSt.decs);
                     nEig = min(size(Coeff, 2), oldM - 1);
                     if nEig > 0, C_con = Coeff(:, 1:nEig)'; end
                catch
                    C_con = [];
                end
            end
            
            % 2. Select Most Promising Directions
            D_con = [];
            if ~isempty(C_con)
                idx = randi(length(OldP_NewEnv));
                x_base = OldP_NewEnv(idx); 
                for i = 1 : size(C_con, 1)
                    D_vec = C_con(i, :);
                    nGen = floor(Problem.N / max(1, (oldM - 1)));
                    Y_decs = [];
                    for k = 1 : nGen
                        y_dec = Algorithm.GenerateSolution(x_base.decs, D_vec, Problem.lower, Problem.upper);
                        Y_decs = [Y_decs; y_dec];
                    end
                    Y = Problem.Evaluation(Y_decs);
                    promising = false;
                    for k = 1 : length(Y)
                        if all(Y(k).objs <= x_base.objs) && any(Y(k).objs < x_base.objs)
                            promising = true; break;
                        end
                    end
                    if promising, D_con = [D_con; D_vec]; end
                end
            end
            
            % 3. Contract PS
            if isempty(D_con)
                NewPop = OldP_NewEnv; return;
            end
            
            nBase = floor(Problem.N / Algorithm.lambda); 
            if nBase < 1, nBase = 1; end
            perm = randperm(length(PSt));
            P_base = PSt(perm(1:min(length(PSt), nBase)));
            
            P_g_Decs = [];
            N_con = size(D_con, 1);
            for i = 1 : length(P_base)
                x_i = P_base(i).decs;
                for j = 1 : N_con
                    D_vec = D_con(j, :);
                    Ng = floor(Problem.N / (N_con * length(P_base)));
                    if Ng < 1, Ng = 1; end
                    for k = 1 : Ng
                        x_new = Algorithm.GenerateSolution(x_i, D_vec, Problem.lower, Problem.upper);
                        P_g_Decs = [P_g_Decs; x_new];
                    end
                end
            end
            P_g = Problem.Evaluation(P_g_Decs);
            
            Combined = [OldP_NewEnv, P_g];
            currentM = size(Combined(1).objs, 2);
            NewPop = Algorithm.DensitySelection(Combined, Problem.N, currentM);
        end
        
        %% Helper Functions
        function x_new = GenerateSolution(~, x, D_vec, lower, upper)
            D_vec(abs(D_vec) < 1e-10) = 1e-10; 
            para = zeros(size(x));
            maskPos = D_vec > 0;
            maskNeg = ~maskPos;
            para(maskPos) = (upper(maskPos) - x(maskPos)) ./ D_vec(maskPos);
            para(maskNeg) = (lower(maskNeg) - x(maskNeg)) ./ D_vec(maskNeg);
            ss = min(para);
            if ss < 0, ss = 0; end
            x_new = x + ss * rand() * D_vec;
            x_new = max(min(x_new, upper), lower);
        end
        
        function Population = DensitySelection(Algorithm, Population, N, M)
            % The "DTAEA Update Mechanism" combined with 
            % "Decomposition-based density estimation"
            
            % 1. Non-dominated Sorting
            [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            
            nNext = sum(Next);
            nLast = length(Last);
            nNeed = N - nNext;
            
            % 2. Decomposition-based Selection
            if nNeed > 0
                LastSols = Population(Last);
                % Use Algorithm.W if it matches M, otherwise re-generate locally
                if size(Algorithm.W, 2) == M
                     W_current = Algorithm.W;
                else
                     [W_current, ~] = Algorithm.UniformPoint(N, M);
                end
                
                Objs = LastSols.objs;
                Zmin = min(Objs, [], 1);
                Zmax = max(Objs, [], 1);
                ObjsNorm = (Objs - Zmin) ./ (Zmax - Zmin + 1e-10);
                
                % Associate solutions with Reference Vectors (Decomposition)
                RegionIdx = zeros(1, nLast);
                for i = 1 : nLast
                    sol = ObjsNorm(i, :);
                    norm_sol = norm(sol);
                    if norm_sol == 0, norm_sol = 1e-6; end
                    cosine = (sol * W_current') / norm_sol;
                    [~, RegionIdx(i)] = max(cosine);
                end
                
                % Niching / Diversity Maintenance
                perm = randperm(nLast);
                RegionIdx = RegionIdx(perm);
                
                % [FIX]: Force unique_idx to be a row vector
                [~, unique_idx] = unique(RegionIdx, 'stable');
                unique_idx = unique_idx(:)'; 
                
                if length(unique_idx) >= nNeed
                    SelectedPermIdx = unique_idx(1:nNeed);
                else
                    SelectedPermIdx = unique_idx;
                    remaining = setdiff(1:nLast, unique_idx);
                    % Both are row vectors now
                    SelectedPermIdx = [SelectedPermIdx, remaining(1:(nNeed - length(unique_idx)))];
                end
                RealSelectedIdx = perm(SelectedPermIdx);
                Next(Last(RealSelectedIdx)) = true;
            end
            Population = Population(Next);
        end
        
        function [W,N] = UniformPoint(~, N, M)
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
