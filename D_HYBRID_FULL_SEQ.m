classdef D_HYBRID_FULL_SEQ < PROBLEM
% <dynamic> <multi/many> <real> <large/none> <expensive/none>
% Dynamic Hybrid Full Sequence:
%   1. Dynamic Objectives (MSeq)
%   2. Dynamic Problem Types (TypeSeq: DTLZ/WFG/ML)
%   3. Dynamic Decision Variables (KSeq)
%
% Problem IDs:
%   1-7   : DTLZ1 - DTLZ7 (Analytic, Negated Output)
%   8-16  : WFG1 - WFG9   (Analytic, Negated Output)
%   17    : Neural Network (NN)
%   18    : Gaussian Process (GPR)
%   19    : Random Forest (RF)
%   20    : GBRT
%
% Parameters:
%   MSeq       --- [2 3 5]    --- Sequence of No. objectives
%   TypeSeq    --- [2 2]   --- Sequence of Problem Types
%   KSeq       --- [20 30 50]    --- Sequence of No. active variables
%   taut       --- 50         --- Generations per stage
%   seed       --- 1          --- Seed for ML models
%   isDiscrete --- 0          --- 0: Continuous, 1: Discrete (25 levels)

    properties(SetAccess = private)
        % Dynamic Sequences
        MSeq
        TypeSeq
        KSeq
        
        taut       % Frequency
        lastT      % Last time step
        
        MaxM       % Max Objectives
        MaxK       % Max Active Variables
        K          % Current Active Variables
        cFix       % Fixed value for inactive variables
        
        % Mapping Logic
        MapProbID
        MapFuncID
        
        % ML Model Parameters
        ML_Params
        seed
        
        % [GUI控制] 离散化开关
        isDiscrete 
    end
    
    methods
        %% Default settings
        function Setting(obj)
            % 1. Parameter parsing
            % Input: MSeq, TypeSeq, KSeq, taut, seed, isDiscrete
            % [GUI修改点] 这里添加第6个参数，PlatEMO会在界面显示第6个输入框
            [obj.MSeq, obj.TypeSeq, obj.KSeq, obj.taut, obj.seed, obj.isDiscrete] = ...
                obj.ParameterSet([2 3 5], [2 2], [20 30 50], 50, 1, 0);
            
            % Normalize vectors
            obj.MSeq = obj.MSeq(:)';
            obj.TypeSeq = obj.TypeSeq(:)';
            obj.KSeq = obj.KSeq(:)';
            
            % Initial State
            obj.MaxM = max(obj.MSeq);
            obj.MaxK = max(obj.KSeq);
            obj.M = obj.MSeq(1);
            obj.K = obj.KSeq(1);
            obj.lastT = 0;
            obj.cFix = 0.1; % Default fixed value for inactive vars
            
            % 2. Variable Dimension Setting
            % D must be large enough to hold MaxK, AND large enough for 
            % the underlying DTLZ/WFG logic (usually M + k_distance).
            % We ensure D is at least MaxK, and at least MaxM + 9.
            obj.D = max(obj.MaxK, obj.MaxM + 9);
            
            % 3. Build Objective Mapping Logic
            obj.MapProbID = zeros(1, obj.MaxM);
            obj.MapFuncID = zeros(1, obj.MaxM);
            UsageCounts = zeros(1, 20); 
            
            typeIdx = 1;
            currentObjIdx = 1;
            
            for k = 1 : length(obj.MSeq)
                targetM = obj.MSeq(k);
                if k == 1, countToAdd = targetM; else, countToAdd = targetM - obj.MSeq(k-1); end
                
                for j = 1 : countToAdd
                    if typeIdx > length(obj.TypeSeq)
                        currentType = obj.TypeSeq(end);
                    else
                        currentType = obj.TypeSeq(typeIdx);
                    end
                    
                    obj.MapProbID(currentObjIdx) = currentType;
                    UsageCounts(currentType) = UsageCounts(currentType) + 1;
                    obj.MapFuncID(currentObjIdx) = UsageCounts(currentType);
                    
                    % Update index logic
                    if k == 1
                        if j == countToAdd, typeIdx = typeIdx + 1; end
                    else
                        typeIdx = typeIdx + 1;
                    end
                    currentObjIdx = currentObjIdx + 1;
                end
            end
            
            % 4. Initialize ML Models
            obj.InitMLModels();
            
            % 5. Bounds
            obj.lower    = zeros(1,obj.D);
            obj.upper    = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        
        %% Initialize ML Parameters
        function InitMLModels(obj)
            rng(obj.seed, 'twister');
            D_total = obj.D; % Models accept full D dimension
            M_total = obj.MaxM; 
            H = 10; 
            
            obj.ML_Params = struct();
            
            % NN (Tanh)
            obj.ML_Params.NN_W1 = randn(M_total, H, D_total);
            obj.ML_Params.NN_b1 = randn(M_total, H);
            obj.ML_Params.NN_W2 = randn(M_total, H);
            obj.ML_Params.NN_b2 = randn(M_total, 1);
            
            % GPR (RBF)
            obj.ML_Params.GPR_C     = rand(M_total, H*2, D_total);
            obj.ML_Params.GPR_alpha = randn(M_total, H*2);
            obj.ML_Params.GPR_l     = 0.2 + 0.3*rand(M_total, 1);
            
            % RF (Step)
            obj.ML_Params.RF_W = randn(M_total, H*2, D_total) * 10;
            obj.ML_Params.RF_b = rand(M_total, H*2) * 2 * pi;
            obj.ML_Params.RF_alpha = randn(M_total, H*2);
            
            % GBRT (ReLU)
            obj.ML_Params.GB_W1 = randn(M_total, H, D_total);
            obj.ML_Params.GB_b1 = randn(M_total, H);
            obj.ML_Params.GB_W2 = randn(M_total, H);
            obj.ML_Params.GB_b2 = randn(M_total, 1);
        end
        
        %% Evaluate solutions
        function Population = Evaluation(obj,varargin)
            % 1. Calculate time stage
            t = floor(obj.FE / obj.N / obj.taut);
            
            % 2. Dynamic Update (M and K independently)
            if t > obj.lastT
                % Update M based on MSeq
                idxM = min(t+1, length(obj.MSeq)); 
                obj.M = obj.MSeq(idxM);
                
                % Update K based on KSeq
                idxK = min(t+1, length(obj.KSeq));
                obj.K = obj.KSeq(idxK);
                
                obj.lastT = t;
            end
            
            % 3. Process Variables
            PopDec = obj.CalDec(varargin{1});
            
            % -------------------------------------------------------------
            % [逻辑] 根据GUI参数决定是否离散化
            % -------------------------------------------------------------
            if obj.isDiscrete > 0
                [N, ~] = size(PopDec);
                num_levels = 25; % 离散值的数量
                
                % 计算每个维度的步长 (Upper - Lower) / 24
                step_size = (obj.upper - obj.lower) / (num_levels - 1);
                
                % 扩展为 N 行以便矩阵运算
                LowerRep = repmat(obj.lower, N, 1);
                StepRep  = repmat(step_size, N, 1);
                UpperRep = repmat(obj.upper, N, 1);

                % 执行离散化：量化到最近的网格点
                PopDec = round((PopDec - LowerRep) ./ StepRep) .* StepRep + LowerRep;
                
                % 边界保护
                PopDec = max(PopDec, LowerRep);
                PopDec = min(PopDec, UpperRep);
            end
            % -------------------------------------------------------------
            
            % [CORE LOGIC]: Mask inactive variables
            % If current K < D, force variables K+1...D to cFix
            if obj.K < obj.D
                PopDec(:, obj.K+1:obj.D) = obj.cFix;
            end
            
            % 4. Calculate Obj & Con
            PopObj = obj.CalObj(PopDec);
            PopCon = obj.CalCon(PopDec);
            
            Population = SOLUTION(PopDec,PopObj,PopCon,...
                                  zeros(size(PopDec,1),1)+obj.FE);
            obj.FE = obj.FE + length(Population);
        end
        
        %% Core Calculation
        function PopObj = CalObj(obj,PopDec)
            [N, ~] = size(PopDec);
            
            % Identify active problem types
            ActiveProbIDs = unique(obj.MapProbID(1:obj.M));
            
            CachedResults = containers.Map('KeyType','double','ValueType','any');
            
            for pid = ActiveProbIDs
                if pid <= 16
                    % Analytic (DTLZ/WFG) -> Negated
                    CachedResults(pid) = GenerateAnalytic(pid, PopDec, obj.MaxM) * (-1);
                else
                    % ML Black-box (17-20) -> Normal [0,1]
                    CachedResults(pid) = obj.GenerateML(pid, PopDec);
                end
            end
            
            % Assemble
            PopObj = zeros(N, obj.M);
            for i = 1 : obj.M
                pid = obj.MapProbID(i);
                fid = obj.MapFuncID(i);
                
                FullM_Result = CachedResults(pid);
                colIdx = mod(fid-1, obj.MaxM) + 1;
                PopObj(:, i) = FullM_Result(:, colIdx);
            end
        end
        
        %% ML Generator
        function FullObj = GenerateML(obj, ProbID, PopDec)
            [N, ~] = size(PopDec);
            M = obj.MaxM;
            FullObj = zeros(N, M);
            
            for m = 1 : M
                switch ProbID
                    case 17 % NN
                        W1 = squeeze(obj.ML_Params.NN_W1(m,:,:));
                        b1 = obj.ML_Params.NN_b1(m,:);
                        W2 = obj.ML_Params.NN_W2(m,:);
                        b2 = obj.ML_Params.NN_b2(m);
                        Z  = PopDec * W1' + repmat(b1, N, 1);
                        val = tanh(Z) * W2' + b2;
                    case 18 % GPR
                        nB = size(obj.ML_Params.GPR_C, 2);
                        C = squeeze(obj.ML_Params.GPR_C(m,:,:));
                        alpha = obj.ML_Params.GPR_alpha(m,:)';
                        ell = obj.ML_Params.GPR_l(m);
                        X_sq = sum(PopDec.^2, 2);
                        C_sq = sum(C.^2, 2)';
                        D_sq = repmat(X_sq, 1, nB) + repmat(C_sq, N, 1) - 2 * (PopDec * C');
                        Phi = exp(-D_sq / (2 * ell^2));
                        val = Phi * alpha;
                    case 19 % RF
                        W = squeeze(obj.ML_Params.RF_W(m,:,:));
                        b = obj.ML_Params.RF_b(m,:);
                        alpha = obj.ML_Params.RF_alpha(m,:)';
                        proj = PopDec * W' + repmat(b, N, 1);
                        val = sign(sin(proj)) * alpha;
                    case 20 % GBRT
                        W1 = squeeze(obj.ML_Params.GB_W1(m,:,:));
                        b1 = obj.ML_Params.GB_b1(m,:);
                        W2 = obj.ML_Params.GB_W2(m,:);
                        b2 = obj.ML_Params.GB_b2(m);
                        Z = PopDec * W1' + repmat(b1, N, 1);
                        val = max(0, Z) * W2' + b2;
                end
                
                v_min = min(val); v_max = max(val);
                if v_max > v_min
                    FullObj(:, m) = (val - v_min) / (v_max - v_min);
                else
                    FullObj(:, m) = val - v_min;
                end
            end
        end
        
        %% Drawing
        function DrawObj(obj,Population)
            Draw(Population.objs,{'f_1','f_2','f_3'});
        end
    end
end

% -------------------------------------------------------------------------
% Analytic Generator (DTLZ & WFG)
% -------------------------------------------------------------------------
function FullObj = GenerateAnalytic(ProbID, PopDec, CalcM)
    [N, D] = size(PopDec);
    
    if ProbID <= 7
        % --- DTLZ 1-7 ---
        XM = PopDec(:,CalcM:D); 
        g = 0;
        FullObj = zeros(N, CalcM);
        
        switch ProbID
            case 1
                g = 100 * (size(XM,2) + sum((XM - 0.5).^2 - cos(20*pi*(XM - 0.5)), 2));
                FullObj = 0.5 * repmat(1+g,1,CalcM) .* fliplr(cumprod([ones(N,1),PopDec(:,1:CalcM-1)],2)) .* [ones(N,1),1-PopDec(:,CalcM-1:-1:1)];
            case 2
                g = sum((XM - 0.5).^2, 2);
                FullObj = repmat(1+g,1,CalcM) .* fliplr(cumprod([ones(N,1),cos(PopDec(:,1:CalcM-1)*pi/2)],2)) .* [ones(N,1),sin(PopDec(:,CalcM-1:-1:1)*pi/2)];
            case 3
                g = 100 * (size(XM,2) + sum((XM - 0.5).^2 - cos(20*pi*(XM - 0.5)), 2));
                FullObj = repmat(1+g,1,CalcM) .* fliplr(cumprod([ones(N,1),cos(PopDec(:,1:CalcM-1)*pi/2)],2)) .* [ones(N,1),sin(PopDec(:,CalcM-1:-1:1)*pi/2)];
            case 4
                PopDec(:,1:CalcM-1) = PopDec(:,1:CalcM-1).^100;
                g = sum((XM - 0.5).^2, 2);
                FullObj = repmat(1+g,1,CalcM) .* fliplr(cumprod([ones(N,1),cos(PopDec(:,1:CalcM-1)*pi/2)],2)) .* [ones(N,1),sin(PopDec(:,CalcM-1:-1:1)*pi/2)];
            case 5
                g = sum((XM - 0.5).^2, 2);
                Theta = repmat(pi/2,N,CalcM-1);
                Theta(:,1) = PopDec(:,1)*pi/2;
                for i = 2 : CalcM-1
                    Theta(:,i) = (1+2*g.*PopDec(:,i))*pi./(4*(1+g));
                end
                FullObj = repmat(1+g,1,CalcM) .* fliplr(cumprod([ones(N,1),cos(Theta)],2)) .* [ones(N,1),sin(Theta(:,CalcM-1:-1:1))];
            case 6
                g = sum(XM.^0.1, 2);
                Theta = repmat(pi/2,N,CalcM-1);
                Theta(:,1) = PopDec(:,1)*pi/2;
                for i = 2 : CalcM-1
                    Theta(:,i) = (1+2*g.*PopDec(:,i))*pi./(4*(1+g));
                end
                FullObj = repmat(1+g,1,CalcM) .* fliplr(cumprod([ones(N,1),cos(Theta)],2)) .* [ones(N,1),sin(Theta(:,CalcM-1:-1:1))];
            case 7
                g = 1 + 9*mean(XM,2);
                FullObj(:,1:CalcM-1) = PopDec(:,1:CalcM-1);
                h = CalcM - sum(FullObj(:,1:CalcM-1)./(1+repmat(g,1,CalcM-1)).*(1+sin(3*pi*FullObj(:,1:CalcM-1))),2);
                FullObj(:,CalcM) = (1+g).*h;
        end
    else
        % --- WFG 1-9 ---
        WFG_ID = ProbID - 7;
        ScaleVec = 2*(1:D);
        z = PopDec .* repmat(ScaleVec, N, 1); 
        FullObj = WFG_Generator(PopDec, WFG_ID, CalcM); 
    end
end

% --- WFG Helper Functions ---
function Obj = WFG_Generator(z, kID, M)
    [N, D] = size(z);
    k = M - 1; 
    l = D - k;
    S = 2*(1:M);
    
    y = z;
    switch kID
        case 1, y(:,k+1:end) = wfg_s_linear(y(:,k+1:end),0.35);
        case 8, y(:,k+1:end) = wfg_b_param(y(:,k+1:end), wfg_r_sum(y(:,1:k),ones(1,k)), 0.98/49.98, 0.02, 50);
        case 9, y(:,1:end-1) = wfg_b_param(y(:,1:end-1), wfg_r_sum(y(:,2:end),ones(1,D-1)), 0.98/49.98, 0.02, 50);
                y(:,end)     = wfg_b_param(y(:,end), wfg_r_sum(y(:,1:end-1),ones(1,D-1)), 0.98/49.98, 0.02, 50);
    end
    switch kID
        case 1, y(:,k+1:end) = wfg_b_flat(y(:,k+1:end),0.8,0.75,0.85);
        case 2, y(:,k+1:end) = wfg_b_flat(wfg_s_linear(y(:,k+1:end),0.35),0.8,0.75,0.85);
        case 3, y(:,k+1:end) = wfg_s_linear(y(:,k+1:end),0.35); 
        case 8, y(:,k+1:end) = wfg_s_linear(y(:,k+1:end),0.35);
        case 9, y(:,1:k)     = wfg_s_decept(y(:,1:k),0.35,0.001,0.05);
                y(:,k+1:end) = wfg_s_multi(y(:,k+1:end),30,10,0.35);
    end
    if ismember(kID, [4,5,6,7,8,9])
        if kID == 4, y = wfg_s_multi(y,30,10,0.35); end
        if kID == 5, y = wfg_s_decept(y,0.35,0.001,0.05); end
    end
    t = zeros(N, M);
    for m = 1 : M-1
        idx_start = (m-1)*k/(M-1) + 1;
        idx_end   = m*k/(M-1);
        t(:,m) = wfg_r_sum(y(:,idx_start:idx_end), ones(1,k/(M-1)));
    end
    t(:,M) = wfg_r_sum(y(:,k+1:end), ones(1,l));
    x = t; 
    h = zeros(N, M);
    for m = 1 : M
        switch kID
            case {1,2} 
                h(:,m) = wfg_convex(x, m, M); 
                if kID==1 && m==M, h(:,m) = wfg_mixed(x,5,1); end 
                if kID==2 && m==M, h(:,m) = wfg_disc(x,5,1,1); end
            case 3 
                h(:,m) = wfg_linear(x, m, M);
            case {4,5,6,7,8,9} 
                h(:,m) = wfg_concave(x, m, M);
        end
    end
    D_val = x(:,M); 
    Obj = repmat(D_val, 1, M) + repmat(S, N, 1) .* h;
end
function y = wfg_s_linear(y,A), y = abs(y-A)./abs(floor(A-y)+A); end
function y = wfg_b_flat(y,A,B,C), min1=min(0,floor(y-B)); min2=min(0,floor(C-y)); y=A+min1.*A.*(B-y)/B-min2.*(1-A).*(y-C)/(1-C); end
function y = wfg_b_param(y,u,A,B,C), v=A-(1-2*u).*abs(floor(0.5-u)+A); y=y.^(B+(C-B)*repmat(v,1,size(y,2))); end
function y = wfg_s_multi(y,A,B,C), tmp1=abs(y-C)./(2*(floor(C-y)+C)); y=(1+cos((4*A+2)*pi*(0.5-tmp1)+2*pi*C))./(4*A+2); end
function y = wfg_s_decept(y,A,B,C), y=1+(abs(y-A)-B).*(floor(y-A+B)*(1-C+(A-B)/B)/(A-B)+floor(A+B-y)*(1-C+(1-A-B)/B)/(1-A-B)+1/B); end
function r = wfg_r_sum(y,w), r=sum(y.*repmat(w,size(y,1),1),2)./sum(w); end
function h = wfg_linear(x, m, M), h=ones(size(x,1),1); for i=1:M-1-m+1, h=h.*x(:,i); end; if m>1, h=h.*(1-x(:,M-m+1)); end; end
function h = wfg_convex(x, m, M), h=ones(size(x,1),1); for i=1:M-1-m+1, h=h.*(1-cos(x(:,i)*pi/2)); end; if m>1, h=h.*(1-sin(x(:,M-m+1)*pi/2)); end; end
function h = wfg_concave(x, m, M), h=ones(size(x,1),1); for i=1:M-1-m+1, h=h.*sin(x(:,i)*pi/2); end; if m>1, h=h.*cos(x(:,M-m+1)*pi/2); end; end
function h = wfg_mixed(x, A, alpha), h=(1-x(:,1)-cos(2*A*pi*x(:,1)+pi/2)/2/A/pi).^alpha; end
function h = wfg_disc(x, A, alpha, beta), h=(1-x(:,1).^(alpha).*cos(A*x(:,1).^beta*pi).^2); end
