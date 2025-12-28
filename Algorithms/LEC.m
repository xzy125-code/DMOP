classdef LEC < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <dynamic>
% Learning to Expand and Contract Pareto Sets
% lambda --- 20 --- Parameter for contraction sampling

%------------------------------- Reference --------------------------------
% G. Ruan, L. L. Minku, S. Menzel, B. Sendhoff, and X. Yao, "Learning to
% Expand/Contract Pareto Sets in Dynamic Multiobjective Optimization With a
% Changing Number of Objectives," IEEE Transactions on Evolutionary
% Computation, 2025.
%--------------------------------------------------------------------------

    properties
        lambda = 20; % Parameter for contraction sampling
    end

    methods
        function main(Algorithm, Problem)
            % Parameter setting
            Algorithm.lambda = Algorithm.ParameterSet(20);
            
            % Initialize Population
            Population = Problem.Initialization();
            
            % Initialize Algorithm State
            lastM = Problem.M;
            
            % Optimization Loop
            while Algorithm.NotTerminated(Population)
                currentM = Problem.M;
                
                % 1. Change Detection & Knowledge Transfer
                if currentM ~= lastM
                    if currentM > lastM
                        % Case: Increasing NObj -> Learning PS Expansion
                        Population = Algorithm.PSExpansion(Population, Problem, lastM);
                    elseif currentM < lastM
                        % Case: Decreasing NObj -> Learning PS Contraction
                        Population = Algorithm.PSContraction(Population, Problem, lastM);
                    end
                    lastM = currentM;
                end
                
                % 2. Evolutionary Optimization Process
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, inf);
                CrowdDis     = CrowdingDistance(Population.objs, FrontNo);
                
                MatingPool = TournamentSelection(2, Problem.N, FrontNo, CrowdDis);
                Offspring  = OperatorGA(Problem, Population(MatingPool));
                
                % Handle Dynamic Change during Offspring Generation
                if size(Offspring(1).objs, 2) ~= size(Population(1).objs, 2)
                    newM = size(Offspring(1).objs, 2);
                    oldM = size(Population(1).objs, 2);
                    
                    if newM > oldM
                        Population = Algorithm.PSExpansion(Population, Problem, oldM);
                    else
                        Population = Algorithm.PSContraction(Population, Problem, oldM);
                    end
                    lastM = newM;
                    
                    [FrontNo, ~] = NDSort(Population.objs, 1);
                    CrowdDis = CrowdingDistance(Population.objs, FrontNo);
                    [~, Rank] = sort(CrowdDis, 'descend');
                    Population = Population(Rank(1:Problem.N));
                else
                    % 3. Environmental Selection
                    Population = Algorithm.EnvironmentalSelection([Population, Offspring], Problem.N);
                end
            end
        end

        %% Part 1: PS Expansion (Learning & Selecting & Expanding)
        function NewPop = PSExpansion(Algorithm, OldPop, Problem, oldM)
            % Extract Old Pareto Set (PSt)
            [FrontNo,~] = NDSort(OldPop.objs, 1);
            PSt = OldPop(FrontNo == 1);
            if isempty(PSt), PSt = OldPop; end
            
            % 1. Learn Candidate Expansion Directions (Algorithm 1)
            Dirs_Cand = [];
            if length(PSt) > 2
                try
                    [Coeff, ~, ~] = pca(PSt.decs);
                    nEig = min(size(Coeff, 2), oldM - 1);
                    if nEig > 0
                        EVs = Coeff(:, 1:nEig); 
                        
                        % Find vectors perpendicular to EVs (Null space)
                        NullBasis = null(EVs'); 
                        
                        % Generate N candidate directions using Latin Hypercube Sampling (LHS)
                        N = Problem.N;
                        nNull = size(NullBasis, 2);
                        
                        if nNull > 0
                            % [Cite: Algorithm 1, Line 4] Use LHS for sampling
                            Coefs = 2 * lhsdesign(N, nNull) - 1; 
                            
                            Dirs_Cand = Coefs * NullBasis';
                            % Normalize
                            len = sqrt(sum(Dirs_Cand.^2, 2));
                            Dirs_Cand = Dirs_Cand ./ (len + 1e-10);
                        end
                    end
                catch
                    Dirs_Cand = [];
                end
            end
            
            % 2. Select Most Promising Directions (Algorithm 3)
            D_exp = [];
            if ~isempty(Dirs_Cand)
                % Re-evaluate PSt in NEW environment for dominance check
                OldP_NewEnv = Problem.Evaluation(PSt.decs);
                
                % [Cite: Algorithm 3, Line 1] "Randomly sample A solution x from OldP"
                % Strict adherence to paper: Use a SINGLE base point for direction SELECTION.
                idx = randi(length(PSt));
                x_base = PSt(idx);
                
                for i = 1 : size(Dirs_Cand, 1)
                    D_vec = Dirs_Cand(i, :);
                    y_dec = Algorithm.GenerateSolution(x_base.decs, D_vec, Problem.lower, Problem.upper);
                    y = Problem.Evaluation(y_dec);
                    
                    % [Cite: Algorithm 3, Line 4] Check if y is non-dominated
                    dominated = false;
                    for k = 1 : length(OldP_NewEnv)
                        if all(OldP_NewEnv(k).objs <= y.objs) && any(OldP_NewEnv(k).objs < y.objs)
                            dominated = true;
                            break;
                        end
                    end
                    
                    if ~dominated
                        D_exp = [D_exp; D_vec];
                    end
                end
            end
            
            % 3. Expand PS (Section III-C)
            % [Modification: Strict Fallback Strategy]
            % [Cite: Section III-C, Paragraph 2] "If no expansion... directions are found... 
            % the population from the old environment is directly copied"
            if isempty(D_exp)
                 % Fallback: Direct Copy (Re-evaluated on new objectives)
                 NewPop = Problem.Evaluation(OldPop.decs);
                 return; 
            end
            
            % If directions found, proceed with Expansion
            N = Problem.N;
            [~, rank] = sort(CrowdingDistance(PSt.objs), 'descend');
            % [Cite: Section III-C-1] "evenly select some solutions... as base solutions"
            BaseSols = PSt(rank(1:min(length(PSt), N)));
            
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
            
            if size(TransferredDecs, 1) > N
                TransferredDecs = TransferredDecs(1:N, :);
            end
            
            NewPop = Problem.Evaluation(TransferredDecs);
        end

        %% Part 2: PS Contraction (Learning & Selecting & Contracting)
        function NewPop = PSContraction(Algorithm, OldPop, Problem, oldM)
            [FrontNo,~] = NDSort(OldPop.objs, 1);
            PSt = OldPop(FrontNo == 1);
            if isempty(PSt), PSt = OldPop; end
            
            % 1. Learn Candidate Contraction Directions (Algorithm 2)
            C_con = [];
            if length(PSt) >= oldM
                try
                     [Coeff, ~, ~] = pca(PSt.decs);
                     nEig = min(size(Coeff, 2), oldM - 1);
                     if nEig > 0
                        C_con = Coeff(:, 1:nEig)'; 
                     end
                catch
                    C_con = [];
                end
            end
            
            % 2. Select Most Promising Directions (Algorithm 4)
            D_con = [];
            if ~isempty(C_con)
                OldP_NewEnv = Problem.Evaluation(PSt.decs);
                
                % [Cite: Algorithm 4, Line 1] "Randomly sample A solution x from OldP"
                % Strict adherence to paper: Use a SINGLE base point for direction SELECTION.
                idx = randi(length(OldP_NewEnv));
                x_base = OldP_NewEnv(idx); 
                
                for i = 1 : size(C_con, 1)
                    D_vec = C_con(i, :);
                    % [Cite: Algorithm 4, Line 3] Generate floor(N/(m-1)) solutions
                    nGen = floor(Problem.N / max(1, (oldM - 1)));
                    Y_decs = [];
                    for k = 1 : nGen
                        y_dec = Algorithm.GenerateSolution(x_base.decs, D_vec, Problem.lower, Problem.upper);
                        Y_decs = [Y_decs; y_dec];
                    end
                    Y = Problem.Evaluation(Y_decs);
                    
                    % [Cite: Algorithm 4, Line 6] Check if any y dominates x (Convergence check)
                    promising = false;
                    for k = 1 : length(Y)
                        if all(Y(k).objs <= x_base.objs) && any(Y(k).objs < x_base.objs)
                            promising = true;
                            break;
                        end
                    end
                    
                    if promising
                        D_con = [D_con; D_vec];
                    end
                end
            end
            
            % 3. Contract PS (Section III-C)
            % [Modification: Strict Fallback Strategy]
            % [Cite: Section III-C] "If no... directions are found... directly copied"
            if isempty(D_con)
                % Fallback: Direct Copy
                NewPop = Problem.Evaluation(OldPop.decs);
                return;
            end
            
            % If directions found, proceed with Contraction
            % Need to ensure OldP_NewEnv is available if we didn't enter the loop above
            if ~exist('OldP_NewEnv', 'var')
                 OldP_NewEnv = Problem.Evaluation(PSt.decs);
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
            NewPop = Algorithm.DensitySelection(Combined, Problem.N);
        end
        
        %% Helper Functions
        function x_new = GenerateSolution(~, x, D_vec, lower, upper)
            D_vec(abs(D_vec) < 1e-10) = 1e-10; 
            
            % [Cite: Equation (3)] Calculate step size 'ss' based on boundary distance
            para = zeros(size(x));
            maskPos = D_vec > 0;
            maskNeg = ~maskPos;
            para(maskPos) = (upper(maskPos) - x(maskPos)) ./ D_vec(maskPos);
            para(maskNeg) = (lower(maskNeg) - x(maskNeg)) ./ D_vec(maskNeg);
            
            ss = min(para);
            if ss < 0, ss = 0; end
            
            % [Cite: Equation (2)] x_new = x + ss * rand() * D
            x_new = x + ss * rand() * D_vec;
            x_new = max(min(x_new, upper), lower);
        end
        
        function Population = EnvironmentalSelection(~, Population, N)
             [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
             Next = FrontNo < MaxFNo;
             Last = find(FrontNo == MaxFNo);
             [~, Rank] = sort(CrowdingDistance(Population(Last).objs), 'descend');
             Next(Last(Rank(1:N-sum(Next)))) = true;
             Population = Population(Next);
        end
        
        function Population = DensitySelection(Algorithm, Population, N)
            [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
            Next = FrontNo < MaxFNo;
            Last = find(FrontNo == MaxFNo);
            
            nNext = sum(Next);
            nLast = length(Last);
            nNeed = N - nNext;
            
            if nNeed > 0
                LastSols = Population(Last);
                M = size(LastSols(1).objs, 2);
                [W, ~] = Algorithm.UniformPoint(N, M);
                Objs = LastSols.objs;
                Zmin = min(Objs, [], 1);
                Zmax = max(Objs, [], 1);
                ObjsNorm = (Objs - Zmin) ./ (Zmax - Zmin + 1e-10);
                
                RegionIdx = zeros(1, nLast);
                for i = 1 : nLast
                    sol = ObjsNorm(i, :);
                    norm_sol = norm(sol);
                    if norm_sol == 0, norm_sol = 1e-6; end
                    cosine = (sol * W') / norm_sol;
                    [~, RegionIdx(i)] = max(cosine);
                end
                
                perm = randperm(nLast);
                RegionIdx = RegionIdx(perm);
                [~, unique_idx] = unique(RegionIdx, 'stable');
                
                if length(unique_idx) >= nNeed
                    SelectedPermIdx = unique_idx(1:nNeed);
                else
                    SelectedPermIdx = unique_idx;
                    remaining = setdiff(1:nLast, unique_idx);
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
