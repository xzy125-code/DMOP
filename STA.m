classdef STA < ALGORITHM
% <multi/many> <real/integer/label/binary/permutation> <dynamic>
% Similarity Transfer Approach (STA)
% -------------------------------------------------------------------------
% Reference:
% G. Ruan, Z. Hou, and X. Yao, "Coping With a Severely Changing Number of
% Objectives in Dynamic Multi-Objective Optimization," IEEE Transactions on
% Evolutionary Computation, 2025.
% -------------------------------------------------------------------------

    properties
        % Archive to store historical populations for different objective numbers
        % Structure: struct('M', {}, 'Pop', {})
        Archive = struct('M', {}, 'Pop', {}); 
        
        % STA specific parameters
        gamma = 0.2;  % Proportion of random solutions for diversity enhancement
        theta = 2;    % Parameter for KTDMOEA expansion
    end

    methods
        function main(Algorithm, Problem)
            % Initialize Population
            Population = Problem.Initialization();
            
            % Initialize Algorithm State
            lastM = Problem.M;
            
            % Save initial population to archive
            Algorithm.UpdateArchive(lastM, Population);
            
            % Optimization Loop
            while Algorithm.NotTerminated(Population)
                currentM = Problem.M;
                
                % 1. Change Detection & Knowledge Transfer
                if currentM ~= lastM
                    % A. Save knowledge from the OLD environment before switching
                    Algorithm.UpdateArchive(lastM, Population);
                    
                    % B. Similarity Transfer: Retrieve & Adapt knowledge
                    Population = Algorithm.SimilarityTransfer(Problem, Population);
                    
                    % Update state
                    lastM = currentM;
                end
                
                % 2. Standard Evolutionary Optimization (NSGA-II based)
                [FrontNo, ~] = NDSort(Population.objs, Population.cons, inf);
                CrowdDis     = CrowdingDistance(Population.objs, FrontNo);
                
                MatingPool = TournamentSelection(2, Problem.N, FrontNo, CrowdDis);
                Offspring  = OperatorGA(Problem, Population(MatingPool));
                
                % Handle Dynamic Change during Offspring Generation
                if size(Offspring(1).objs, 2) ~= size(Population(1).objs, 2)
                    newM = size(Offspring(1).objs, 2);
                    oldM = size(Population(1).objs, 2);
                    
                    Algorithm.UpdateArchive(oldM, Population);
                    Population = Algorithm.SimilarityTransfer(Problem, Population); 
                    lastM = newM;
                else
                    Population = Algorithm.EnvironmentalSelection([Population, Offspring], Problem.N);
                end
            end
        end

        %% Method: Update Archive
        function UpdateArchive(Algorithm, M, Pop)
            [FrontNo, ~] = NDSort(Pop.objs, 1);
            BestSols = Pop(FrontNo == 1);
            if isempty(BestSols), BestSols = Pop; end
            
            if isempty(Algorithm.Archive)
                idx = [];
            else
                idx = find([Algorithm.Archive.M] == M, 1);
            end
            
            if isempty(idx)
                newIdx = length(Algorithm.Archive) + 1;
                Algorithm.Archive(newIdx).M = M;
                Algorithm.Archive(newIdx).Pop = BestSols; 
            else
                Algorithm.Archive(idx).Pop = BestSols;
            end
        end
        
        %% Method: Similarity Transfer Strategy (The Core of STA)
        function NewPop = SimilarityTransfer(Algorithm, Problem, ~)
            currentM = Problem.M;
            
            if isempty(Algorithm.Archive)
                NewPop = Problem.Initialization();
                return;
            end
            
            HistoryMs = [Algorithm.Archive.M];
            Distances = abs(HistoryMs - currentM);
            minDist = min(Distances);
            candidates = find(Distances == minDist);
            
            % Select the first candidate
            bestIdx = candidates(1); 
            
            SourceEnv = Algorithm.Archive(bestIdx);
            SourcePop = SourceEnv.Pop;
            M_tr = SourceEnv.M;
            M_c  = currentM;
            
            TransferredDecs = [];
            
            if M_tr < M_c
                % Case A: Increasing NObj -> Use KTDMOEA Expansion + STA Randomization
                [TransferredDecs, N_exp] = Algorithm.KTDMOEA_Expansion(SourcePop, Problem, M_tr);
                
                % STA Strategy: Randomization Enhancing Diversity
                threshold = (M_c - M_tr) / 2;
                if N_exp < threshold
                    N = Problem.N;
                    nRandom = round(Algorithm.gamma * N);
                    
                    if nRandom > 0
                        RandomDecs = repmat(Problem.lower, nRandom, 1) + ...
                                     rand(nRandom, Problem.D) .* (repmat(Problem.upper, nRandom, 1) - repmat(Problem.lower, nRandom, 1));
                        
                        if size(TransferredDecs, 1) >= N
                             TransferredDecs(end-nRandom+1:end, :) = RandomDecs;
                        else
                             TransferredDecs = [TransferredDecs; RandomDecs];
                        end
                    end
                end
                
            elseif M_tr > M_c
                % Case B: Decreasing NObj -> Use LEC Contraction
                TransferredDecs = Algorithm.LEC_Contraction(SourcePop, Problem, M_tr);
                
            else
                % Case C: Same NObj -> Direct Copy
                TransferredDecs = SourcePop.decs;
                TransferredDecs = Algorithm.AdjustDecsDimension(TransferredDecs, Problem.D, Problem.lower, Problem.upper);
            end
            
            N = Problem.N;
            [nGen, dGen] = size(TransferredDecs);
            
            if nGen < N
                nNeed = N - nGen;
                RandomDecs = repmat(Problem.lower, nNeed, 1) + ...
                             rand(nNeed, dGen) .* (repmat(Problem.upper, nNeed, 1) - repmat(Problem.lower, nNeed, 1));
                TransferredDecs = [TransferredDecs; RandomDecs];
            elseif nGen > N
                perm = randperm(nGen);
                TransferredDecs = TransferredDecs(perm(1:N), :);
            end
            
            NewPop = Problem.Evaluation(TransferredDecs);
        end
        
        %% Helper 1: KTDMOEA Expansion Logic
        function [NewDecs, N_exp] = KTDMOEA_Expansion(Algorithm, SourcePop, Problem, M_tr)
            % 1. Identify Expansion Directions
            [Dirs, ~] = Algorithm.GetExpansionDirs(SourcePop, Problem);
            N_exp = size(Dirs, 1);
            
            if N_exp == 0
                Dirs = rand(1, Problem.D); 
                N_exp = 0; 
            end
            
            % 2. Generate Solutions along Directions
            N = Problem.N;
            N_dir = size(Dirs, 1);
            N_base = floor((N - M_tr) / (N_dir * Algorithm.theta)); 
            if N_base < 1, N_base = 1; end
            
            [~, rank] = sort(CrowdingDistance(SourcePop.objs), 'descend');
            BaseSols = SourcePop(rank(1:min(length(SourcePop), N)));
            
            NewDecs = [];
            BaseDecs = Algorithm.AdjustDecsDimension(BaseSols.decs, Problem.D, Problem.lower, Problem.upper);
            
            for i = 1 : size(BaseDecs, 1)
                x = BaseDecs(i, :);
                for j = 1 : N_dir
                    D_vec = Dirs(j, :);
                    for k = 1 : Algorithm.theta
                        x_new = Algorithm.GenerateSolution(x, D_vec, Problem.lower, Problem.upper);
                        NewDecs = [NewDecs; x_new];
                    end
                end
                if size(NewDecs, 1) >= N
                    break;
                end
            end
            
            if size(NewDecs, 1) < N
                nRem = N - size(NewDecs, 1);
                FillDecs = BaseDecs(1:min(nRem, size(BaseDecs,1)), :);
                if size(FillDecs, 1) < nRem
                     nNeed = nRem - size(FillDecs, 1);
                     RandFill = repmat(Problem.lower, nNeed, 1) + ...
                                rand(nNeed, Problem.D) .* (repmat(Problem.upper, nNeed, 1) - repmat(Problem.lower, nNeed, 1));
                     FillDecs = [FillDecs; RandFill];
                end
                NewDecs = [NewDecs; FillDecs];
            end
        end
        
        %% Helper 2: LEC Contraction Logic
        function NewDecs = LEC_Contraction(Algorithm, SourcePop, Problem, M_tr)
            BaseDecs = Algorithm.AdjustDecsDimension(SourcePop.decs, Problem.D, Problem.lower, Problem.upper);
            
            C_con = [];
            try
                [Coeff, ~, ~] = pca(BaseDecs);
                nEig = min(size(Coeff, 2), M_tr - 1);
                if nEig > 0
                    C_con = Coeff(:, 1:nEig)';
                end
            catch
            end
            
            if isempty(C_con)
                 C_con = 2*rand(1, Problem.D) - 1;
            end
            
            % [ADDED] Selection of Most Promising Contraction Directions (Algorithm 4 in LEC)
            % This was missing in the previous version
            D_con = [];
            SourceInTarget = Problem.Evaluation(BaseDecs);
            idx = randi(length(SourceInTarget));
            x_base = SourceInTarget(idx); 
            
            for i = 1 : size(C_con, 1)
                D_vec = C_con(i, :);
                % Determine if direction helps convergence
                nGen = floor(Problem.N / max(1, (M_tr - 1)));
                Y_decs = [];
                for k = 1 : nGen
                    y_dec = Algorithm.GenerateSolution(x_base.decs, D_vec, Problem.lower, Problem.upper);
                    Y_decs = [Y_decs; y_dec];
                end
                
                if isempty(Y_decs), continue; end
                Y = Problem.Evaluation(Y_decs);
                
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
            
            if isempty(D_con), D_con = C_con; end
            
            % Execution
            lambda_param = floor(Problem.N / 20); 
            if lambda_param < 1, lambda_param = 1; end
            
            perm = randperm(size(BaseDecs, 1));
            P_base = BaseDecs(perm(1:min(size(BaseDecs, 1), lambda_param)), :);
            
            NewDecs = [];
            N_con = size(D_con, 1);
            Ng = floor(Problem.N / (N_con * size(P_base, 1)));
            if Ng < 1, Ng = 1; end
            
            for i = 1 : size(P_base, 1)
                x_i = P_base(i, :);
                for j = 1 : N_con
                    D_vec = D_con(j, :);
                    for k = 1 : Ng
                         x_new = Algorithm.GenerateSolution(x_i, D_vec, Problem.lower, Problem.upper);
                         NewDecs = [NewDecs; x_new];
                    end
                end
            end
            
            if size(NewDecs, 1) < Problem.N
                 nRem = Problem.N - size(NewDecs, 1);
                 indices = randi(size(BaseDecs, 1), nRem, 1);
                 Fill = BaseDecs(indices, :);
                 NewDecs = [NewDecs; Fill];
            end
        end
        
        %% Utility: Find Expansion Directions (Consistent with KTDMOEA Paper)
        function [Dirs, Valid] = GetExpansionDirs(Algorithm, SourcePop, Problem)
             Dirs = [];
             Valid = false;
             
             % 1. Prepare Data
             D = Problem.D;
             lower = Problem.lower;
             upper = Problem.upper;
             
             SourceDecs = Algorithm.AdjustDecsDimension(SourcePop.decs, D, lower, upper);
             SourceInTarget = Problem.Evaluation(SourceDecs);
             
             % 2. Find Extreme Points
             ObjsOrigin = SourcePop.objs; 
             ExtremePointsDecs = [];
             
             for i = 1 : size(ObjsOrigin, 2)
                 [~, rank] = sort(ObjsOrigin(:, i), 'descend');
                 ExtremePointsDecs = [ExtremePointsDecs; SourceDecs(rank(1), :)];
             end
             ExtremePointsDecs = unique(ExtremePointsDecs, 'rows');
             
             % 3. Generate Scout Solutions (P_var) via Mutation
             P_var_Decs = [];
             Xe_List = [];
             
             if size(lower, 1) > 1, lower = lower'; end
             if size(upper, 1) > 1, upper = upper'; end
             
             for i = 1 : size(ExtremePointsDecs, 1)
                 xe = ExtremePointsDecs(i, :);
                 
                 N_mutants = 100;
                 Mus = repmat(xe, N_mutants, 1);
                 
                 LowerRep = repmat(lower, N_mutants, 1);
                 UpperRep = repmat(upper, N_mutants, 1);
                 
                 Site = rand(size(Mus)) < 1/D;
                 mu = rand(size(Mus));
                 
                 temp = Site & mu<=0.5;
                 Mus(temp) = Mus(temp)+(UpperRep(temp)-LowerRep(temp)).*((2.*mu(temp)+(1-2.*mu(temp)).*...
                             (1-(Mus(temp)-LowerRep(temp))./(UpperRep(temp)-LowerRep(temp))).^21).^(1/21)-1);
                 
                 temp = Site & mu>0.5;
                 Mus(temp) = Mus(temp)+(UpperRep(temp)-LowerRep(temp)).*(1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).*...
                             (1-(UpperRep(temp)-Mus(temp))./(UpperRep(temp)-LowerRep(temp))).^21).^(1/21));
                 
                 Mus = max(min(Mus, UpperRep), LowerRep);
                 
                 P_var_Decs = [P_var_Decs; Mus];
                 Xe_List = [Xe_List; repmat(xe, N_mutants, 1)]; 
             end
             
             if isempty(P_var_Decs)
                 return;
             end
             
             % 4. Evaluate Scout Solutions in Target Environment
             P_var_Pop = Problem.Evaluation(P_var_Decs);
             
             % 5. Filter Dominated Solutions
             IsDominated = false(length(P_var_Pop), 1);
             TargetObjs = SourceInTarget.objs;
             ScoutObjs  = P_var_Pop.objs;
             
             for i = 1 : length(P_var_Pop)
                 currentObj = ScoutObjs(i, :);
                 dominators = all(TargetObjs <= currentObj, 2) & any(TargetObjs < currentObj, 2);
                 if any(dominators)
                     IsDominated(i) = true;
                 end
             end
             
             % 6. [ADDED] Density Estimation (Occupied Regions) Check
             % This was missing in the previous version but is required by KTDMOEA
             KeepIdx = find(~IsDominated);
             
             if ~isempty(KeepIdx)
                 % Generate Weight Vectors based on CURRENT objective number
                 currentM = size(TargetObjs, 2);
                 [W, ~] = Algorithm.UniformPoint(Problem.N, currentM);
                 
                 % Normalize all known objectives (Source + P_var) to determine regions
                 AllObjs = [TargetObjs; ScoutObjs(KeepIdx, :)];
                 Zmin = min(AllObjs, [], 1);
                 Zmax = max(AllObjs, [], 1);
                 Zmax(Zmax == Zmin) = Zmax(Zmax == Zmin) + 1e-6;
                 
                 % Identify occupied regions by Source Solutions
                 OccupiedRegions = false(1, size(W, 1));
                 SourceNorm = (TargetObjs - Zmin) ./ (Zmax - Zmin);
                 
                 for i = 1 : size(SourceNorm, 1)
                     norm_sol = norm(SourceNorm(i, :));
                     if norm_sol == 0, continue; end
                     cosine = (SourceNorm(i, :) * W') / norm_sol;
                     [~, regionIdx] = max(cosine);
                     OccupiedRegions(regionIdx) = true;
                 end
                 
                 % Filter P_var based on Occupied Regions
                 FinalKeep = false(size(KeepIdx));
                 ScoutNorm = (ScoutObjs(KeepIdx, :) - Zmin) ./ (Zmax - Zmin);
                 
                 for i = 1 : length(KeepIdx)
                     norm_sol = norm(ScoutNorm(i, :));
                     if norm_sol == 0 
                         FinalKeep(i) = true; 
                         continue; 
                     end
                     cosine = (ScoutNorm(i, :) * W') / norm_sol;
                     [~, regionIdx] = max(cosine);
                     
                     % Keep if region is NOT occupied
                     if ~OccupiedRegions(regionIdx)
                         FinalKeep(i) = true;
                     end
                 end
                 
                 ValidIdx = KeepIdx(FinalKeep);
             else
                 ValidIdx = [];
             end
             
             % 7. Form Directions
             if ~isempty(ValidIdx)
                 ValidPVar = P_var_Decs(ValidIdx, :);
                 ValidXe   = Xe_List(ValidIdx, :);
                 
                 RawDirs = ValidPVar - ValidXe;
                 len = sqrt(sum(RawDirs.^2, 2));
                 Dirs = RawDirs ./ (len + 1e-10);
                 
                 Dirs = unique(Dirs, 'rows');
                 Valid = true;
             end
        end
        
        %% Utility: Generate Solution with Step Size
        function x_new = GenerateSolution(~, x, D_vec, lower, upper)
            D_vec(abs(D_vec) < 1e-10) = 1e-10; 
            
            para = zeros(size(x));
            maskPos = D_vec > 0;
            maskNeg = ~maskPos;
            
            if isscalar(lower), lower = repmat(lower, size(x)); end
            if isscalar(upper), upper = repmat(upper, size(x)); end
            
            para(maskPos) = (upper(maskPos) - x(maskPos)) ./ D_vec(maskPos);
            para(maskNeg) = (lower(maskNeg) - x(maskNeg)) ./ D_vec(maskNeg);
            
            ss = min(para);
            if ss < 0, ss = 0; end 
            
            x_new = x + ss * rand() * D_vec;
            x_new = max(min(x_new, upper), lower);
        end
        
        %% Utility: Adjust Dimension
        function NewDecs = AdjustDecsDimension(~, OldDecs, TargetD, lower, upper)
            [N, OldD] = size(OldDecs);
            NewDecs = zeros(N, TargetD);
            minD = min(OldD, TargetD);
            NewDecs(:, 1:minD) = OldDecs(:, 1:minD);
            
            if isscalar(lower), lower = repmat(lower, 1, TargetD); end
            if isscalar(upper), upper = repmat(upper, 1, TargetD); end
            
            if TargetD > OldD
                 low_exp = repmat(lower(OldD+1:end), N, 1);
                 upp_exp = repmat(upper(OldD+1:end), N, 1);
                 NewDecs(:, OldD+1:end) = low_exp + rand(N, TargetD-OldD) .* (upp_exp - low_exp);
            end
        end
        
        %% Helper: Environmental Selection (NSGA-II)
        function Population = EnvironmentalSelection(~, Population, N)
             [FrontNo, MaxFNo] = NDSort(Population.objs, Population.cons, N);
             Next = FrontNo < MaxFNo;
             Last = find(FrontNo == MaxFNo);
             [~, Rank] = sort(CrowdingDistance(Population(Last).objs), 'descend');
             Next(Last(Rank(1:N-sum(Next)))) = true;
             Population = Population(Next);
        end

        %% Helper: UniformPoint (Copied from KTDMOEA/LEC for consistency)
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