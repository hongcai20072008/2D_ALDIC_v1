% ---------------------------------------------
% Augmented Lagrangian Digital Image Correlation (AL-DIC)
% working with an adaptive quadtree mesh
% 
% Author: Jin Yang, PhD @Caltech
% Contact and support: jyang526@wisc.edu -or- aldicdvc@gmail.com
% Date: 2015.04,06,07; 2016.03,04; 2020.11
% ---------------------------------------------

%% Section 1: Clear MATLAB environment & mex set up Spline interpolation  
close all; clear; clc; clearvars -global
fprintf('------------ Section 1 Start ------------ \n')
setenv('MW_MINGW64_LOC','C:\TDM-GCC-64');
try mex -O ba_interp2.cpp; catch; end % mex set up ba_interp2.cpp script
% [Comment]: If this line reports error but it works before, 
% Change line 14 to: "try mex -O ba_interp2.cpp; catch; end"
addpath("./func",'./func/rbfinterp','./plotFiles','./func_quadtree','./func_quadtree/refinement','./plotFiles/export_fig-d966721/'); 
% TODO: addpath("./YOUR IMAGE FOLDER"); 
fprintf('------------ Section 1 Done ------------ \n \n')


%% Section 2: Load DIC parameters and set up DIC parameters
fprintf('------------ Section 2 Start ------------ \n')
% ====== Read images ======
[file_name,Img,DICpara] = ReadImage; % Load DIC raw images
% %%%%%% Uncomment the line below to change the DIC computing region (ROI) manually %%%%%%
% DICpara.gridxROIRange = [gridxROIRange1,gridxROIRange2]; DICpara.gridyROIRange = [Val1, Val2];
% E.g., gridxROIRange = [224,918]; gridyROIRange = [787,1162];

% ====== Normalize images: fNormalized = (f-f_avg)/(f_std) ======
[ImgNormalized,DICpara.gridxyROIRange] = funNormalizeImg(Img,DICpara.gridxyROIRange);
fNormalized = ImgNormalized{1}; % Load the first referece image

% ====== Compute image gradients ======
Df = funImgGradient(fNormalized,fNormalized); % Finite difference to compute image grayscale gradients;

% ====== Initialize variable storage ======
ResultDisp = cell(length(ImgNormalized)-1,1);    ResultDefGrad = cell(length(ImgNormalized)-1,1);
ResultStrainWorld = cell(length(ImgNormalized)-1,1);  ResultStressWorld = cell(length(ImgNormalized)-1,1);
ResultFEMeshEachFrame = cell(length(ImgNormalized)-1,1); % Needs future improvment: to store FE-mesh for each frame
ResultFEMesh = cell(ceil((length(ImgNormalized)-1)/DICpara.ImgSeqIncUnit),1); % For incremental DIC mode
fprintf('------------ Section 2 Done ------------ \n \n')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To solve each frame in an image sequence
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ImgSeqNum = 2 : length(ImgNormalized)  
    
    disp(['Current image frame #: ', num2str(ImgSeqNum),'/',num2str(length(ImgNormalized))]);
    gNormalized = ImgNormalized{ImgSeqNum}; % Load current deformed image frame 
     
    
    %% Section 3: Compute an initial guess of the unknown displacement field
    fprintf('\n'); fprintf('------------ Section 3 Start ------------ \n')
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is to find or update an initial guess of the unknown displacements.
    % The key idea is to either to use a new FFT-based cross correlation peak fitting,
    % or use the results from previous frames to compute a new initial guess for the next frame;
    % Particularly in the incremental mode DIC, the reference image can also be updated, e.g.,
    % " fNormalized = ImgNormalized{ImgSeqNum-mod(ImgSeqNum-1,ImgSeqIncUnit)}; "
    %
    % DICpara.NewFFTSearch = 0; % If you want to apply the FFT-based cross correlation to 
    % compute the initial guess for each frame, please make sure that "DICpara.NewFFTSearch = 0". 
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%% One practical strategy is to let first 7 frames do the FFT-based 
    %%%%% cross correlation and then using the data driven method to estimate  
    %%%%% initial guesses for other frames  
    if ImgSeqNum == 2
        DICpara.NewFFTSearch = 1; DICpara.InitFFTSearchMethod = [];
    elseif ImgSeqNum < 7 || ImgSeqNum > DICpara.ImgSeqIncUnit
        DICpara.NewFFTSearch = 1; % Use FFT-based cross correlation to compute the initial guess
    else
        DICpara.NewFFTSearch = 0; % Apply data driven method to estimate initial guesses for later frames
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ImgSeqNum == 2 || (DICpara.NewFFTSearch == 1 && mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit) ~= 0) % Apply FFT-based cross correlation to compute the initial guess 
        
        % ====== Integer Search ======
        [DICpara,x0temp,y0temp,u,v,cc]= IntegerSearch(fNormalized,gNormalized,file_name,DICpara);

        % %%%%% Optional codes to measure more gridded measurement points %%%%%
        % [DICpara,x0temp_f,y0temp_f,u_f,v_f,cc]= IntegerSearch(fNormalized,gNormalized,file_name,DICpara);
        % 
        % xnodes = max([1+0.5*DICpara.winsize+ DICpara.SizeOfFFTSearchRegion(1), DICpara.gridxyROIRange.gridx(1) ])  ...
        %     : DICpara.winstepsize : min([size(fNormalized,1)-0.5*DICpara.winsize-1- DICpara.SizeOfFFTSearchRegion(1),DICpara.gridxyROIRange.gridx(2) ]);
        % ynodes = max([1+0.5*DICpara.winsize+ DICpara.SizeOfFFTSearchRegion(2),DICpara.gridxyROIRange.gridy(1) ])  ...
        %     : DICpara.winstepsize : min([size(fNormalized,2)-0.5*DICpara.winsize-1- DICpara.SizeOfFFTSearchRegion(2),DICpara.gridxyROIRange.gridy(2) ]);
        %  
        % [x0temp,y0temp] = ndgrid(xnodes,ynodes);   u_f_NotNanInd = find(~isnan(u_f(:)));
        %  
        % op1 = rbfcreate( [x0temp_f(u_f_NotNanInd),y0temp_f(u_f_NotNanInd)]',[u_f(u_f_NotNanInd)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        % u = rbfinterp( [x0temp(:),y0temp(:)]', op1 );
        % op2 = rbfcreate( [x0temp_f(u_f_NotNanInd),y0temp_f(u_f_NotNanInd)]',[v_f(u_f_NotNanInd)]','RBFFunction', 'thinplate'); %rbfcheck(op2);
        % v = rbfinterp([x0temp(:),y0temp(:)]', op2 );
        % x0temp = x0temp'; y0temp = y0temp'; u=u'; v=v';
        
        % %%%%% Do some regularization to further decrease the noise %%%%%
        % % u = regularizeNd([x0temp(:),y0temp(:)],u(:),{xnodes',ynodes'},1e-3);
        % % v = regularizeNd([x0temp(:),y0temp(:)],v(:),{xnodes',ynodes'},1e-3);
        
        % ====== FEM mesh set up ======
        [DICmesh] = MeshSetUp(x0temp,y0temp,DICpara); clear x0temp y0temp;
        % ====== Initial Value ======
        U0 = Init(u,v,cc.max,DICmesh.x0,DICmesh.y0,0); % PlotuvInit; [x0temp,y0temp,u,v,cc]= IntegerSearchMg(fNormalized,gNormalized,file_name,DICpara);
        % ====== Deal with incremental mode ======
        fNormalizedNewIndex = ImgSeqNum-mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit)-1;
        if DICpara.ImgSeqIncUnit == 1, fNormalizedNewIndex = fNormalizedNewIndex-1; end
        ResultFEMesh{1+floor(fNormalizedNewIndex/DICpara.ImgSeqIncUnit)} = ... % To save first mesh info
            struct( 'coordinatesFEM',DICmesh.coordinatesFEM,'elementsFEM',DICmesh.elementsFEM, ...
            'winsize',DICpara.winsize,'winstepsize',DICpara.winstepsize,'gridxyROIRange',DICpara.gridxyROIRange );
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    elseif mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit) == 0 % To update ref image in incremental mode
        fNormalizedNewIndex = ImgSeqNum-1; 
        fNormalized = ImgNormalized{fNormalizedNewIndex}; % Update reference
        Df = funImgGradient(fNormalized,fNormalized);
        [DICpara,~] = ReadImageRefUpdate(file_name,ImgSeqNum,ResultDisp{ImgSeqNum-2}.U,DICpara,DICmesh); % Update reference image if needed;
        [DICpara,x0temp_f,y0temp_f,u_f,v_f,cc]= IntegerSearch(fNormalized,gNormalized,file_name,DICpara);
        
        % ====== FEM mesh set up ======
        xnodes = max([1+0.5*DICpara.winsize+ DICpara.SizeOfFFTSearchRegion(1), DICpara.gridxyROIRange.gridx(1) ])  ...
            : DICpara.winstepsize : min([size(fNormalized,1)-0.5*DICpara.winsize-1- DICpara.SizeOfFFTSearchRegion(1),DICpara.gridxyROIRange.gridx(2) ]);
        ynodes = max([1+0.5*DICpara.winsize+ DICpara.SizeOfFFTSearchRegion(2),DICpara.gridxyROIRange.gridy(1) ])  ...
            : DICpara.winstepsize : min([size(fNormalized,2)-0.5*DICpara.winsize-1- DICpara.SizeOfFFTSearchRegion(2),DICpara.gridxyROIRange.gridy(2) ]);
         
        [x0temp,y0temp] = ndgrid(xnodes,ynodes);   u_f_NotNanInd = find(~isnan(u_f(:)));
         
        op1 = rbfcreate( [x0temp_f(u_f_NotNanInd),y0temp_f(u_f_NotNanInd)]',[u_f(u_f_NotNanInd)]','RBFFunction', 'thinplate'); %rbfcheck(op1);
        u = rbfinterp( [x0temp(:),y0temp(:)]', op1 );
        op2 = rbfcreate( [x0temp_f(u_f_NotNanInd),y0temp_f(u_f_NotNanInd)]',[v_f(u_f_NotNanInd)]','RBFFunction', 'thinplate'); %rbfcheck(op2);
        v = rbfinterp([x0temp(:),y0temp(:)]', op2 );
        x0temp = x0temp'; y0temp = y0temp'; u=u'; v=v';
        
        %%%%% Do some regularization to further decrease the noise %%%%%
        % u = regularizeNd([x0temp(:),y0temp(:)],u(:),{xnodes',ynodes'},1e-3);
        % v = regularizeNd([x0temp(:),y0temp(:)],v(:),{xnodes',ynodes'},1e-3);
        
        [DICmesh] = MeshSetUp(x0temp,y0temp,DICpara); clear x0temp y0temp;
        % ====== Initial Value ======
        U0 = Init(u,v,cc.max,DICmesh.x0,DICmesh.y0,0); 
        ResultFEMesh{1+floor(fNormalizedNewIndex/DICpara.ImgSeqIncUnit)} = ... % To save first mesh info
            struct( 'coordinatesFEM',DICmesh.coordinatesFEM,'elementsFEM',DICmesh.elementsFEM, ...
            'winsize',DICpara.winsize,'winstepsize',DICpara.winstepsize,'gridxyROIRange',DICpara.gridxyROIRange );
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else % Use the solved results from the last frame as the new initial guess
        if ImgSeqNum < 7 % Import previous U for ImgSeqNum [2,6] 
            U0 = ResultDisp{ImgSeqNum-2}.U;
             
        else % When ImgSeqNum > 6: POD predicts next disp U0 from previous results of (ImgSeqNum+[-5:1:-1])
            nTime = 5; np = length(ResultDisp{ImgSeqNum-2}.U)/2; % "nTime" value 5 is an empirical value, can be changed.
            T_data_u = zeros(nTime,np); T_data_v = zeros(nTime,np); 
            for tempi = 1:nTime
                T_data_u(tempi,:) = ResultDisp{ImgSeqNum-(2+nTime)+tempi, 1}.U(1:2:np*2)';
                T_data_v(tempi,:) = ResultDisp{ImgSeqNum-(2+nTime)+tempi, 1}.U(2:2:np*2)';
        
            end
            nB = 3; t_train = [ImgSeqNum-1-nTime:ImgSeqNum-2]'; t_pre = [ImgSeqNum-1]';
            [u_pred,~,~,~] = funPOR_GPR(T_data_u,t_train,t_pre,nB);
            [v_pred,~,~,~] = funPOR_GPR(T_data_v,t_train,t_pre,nB);
            tempu = u_pred(1,:); tempv = v_pred(1,:);
            U0 = [tempu(:),tempv(:)]'; U0 = U0(:);
         
            % %%%%% After running the new ImgSeqNum, you can uncomment these 
            % %%%%% lines to compare how the initial guess has been improved.  
            % Plotdisp_show(U0-ResultDisp{ImgSeqNum-1}.U,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4));
            % Plotdisp_show(ResultDisp{ImgSeqNum-2}.U-ResultDisp{ImgSeqNum-1}.U,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM(:,1:4));
        end
    end
    
    % ====== Compute f(X)-g(x+u) ======
    % PlotImgDiff(x0,y0,u,v,fNormalized,gNormalized);
    ResultFEMeshEachFrame{ImgSeqNum-1} = struct( 'coordinatesFEM',DICmesh.coordinatesFEM,'elementsFEM',DICmesh.elementsFEM);
    fprintf('------------ Section 3 Done ------------ \n \n')
    
    
    %% Section 4: Subproblem 1 -or- Local ICGN Subset DIC
    fprintf('------------ Section 4 Start ------------ \n')
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is to solve the first local step in ALDIC: Subproblem 1
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % ====== ALStep 1 Subproblem1: Local Subset DIC ======
    mu=0; beta=0; tol=1e-2; ALSolveStep=1; ALSub1Time=zeros(6,1); ALSub2Time=zeros(6,1); 
    ConvItPerEle=zeros(size(DICmesh.coordinatesFEM,1),6); ALSub1BadPtNum=zeros(6,1);
    disp(['***** Start step',num2str(ALSolveStep),' Subproblem1 *****'])
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ------ Start Local DIC IC-GN iteration ------
    [USubpb1,FSubpb1,HtempPar,ALSub1Timetemp,ConvItPerEletemp,LocalICGNBadPtNumtemp] = ...
        LocalICGN(U0,DICmesh.coordinatesFEM,Df,fNormalized,gNormalized,DICpara,'GaussNewton',tol);
    ALSub1Time(ALSolveStep) = ALSub1Timetemp; ConvItPerEle(:,ALSolveStep) = ConvItPerEletemp; ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp; toc
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ------  Manually find some bad points from Local Subset ICGN step ------
    % Comment these lines below if you don't need to manually remove local bad points
    % %%%%% Comment START %%%%%   
        close all; USubpb1World = USubpb1; USubpb1World(2:2:end) = -USubpb1(2:2:end);
        Plotuv(USubpb1World,DICmesh.x0,DICmesh.y0World);
        disp('--- Start to manually remove bad points ---')
        u = reshape(USubpb1(1:2:end),size(DICmesh.x0,1),size(DICmesh.x0,2)); 
        v = reshape(USubpb1(2:2:end),size(DICmesh.x0,1),size(DICmesh.x0,2));
        [u,v,~,Local_BadptRow,Local_BadptCol,RemoveOutliersList] = funRemoveOutliers(u',v',[],0.5,100); u=u';v=v';
        USubpb1(1:2:end) = reshape(u,size(DICmesh.coordinatesFEM,1),1); USubpb1(2:2:end) = reshape(v,size(DICmesh.coordinatesFEM,1),1);
        f11 = reshape(FSubpb1(1:4:end),size(DICmesh.x0,1),size(DICmesh.x0,2)); 
        f21 = reshape(FSubpb1(2:4:end),size(DICmesh.x0,1),size(DICmesh.x0,2));
        f12 = reshape(FSubpb1(3:4:end),size(DICmesh.x0,1),size(DICmesh.x0,2)); 
        f22 = reshape(FSubpb1(4:4:end),size(DICmesh.x0,1),size(DICmesh.x0,2));
        f11=f11'; f11(RemoveOutliersList) = NaN; f11 = inpaint_nans(f11,4); f11=f11';  
        f21=f21'; f21(RemoveOutliersList) = NaN; f21 = inpaint_nans(f21,4); f21=f21'; 
        f12=f12'; f12(RemoveOutliersList) = NaN; f12 = inpaint_nans(f12,4); f12=f12'; 
        f22=f22'; f22(RemoveOutliersList) = NaN; f22 = inpaint_nans(f22,4); f22=f22';
        FSubpb1(1:4:end) = reshape(f11,size(DICmesh.coordinatesFEM,1),1); 
        FSubpb1(2:4:end) = reshape(f21,size(DICmesh.coordinatesFEM,1),1);
        FSubpb1(3:4:end) = reshape(f12,size(DICmesh.coordinatesFEM,1),1); 
        FSubpb1(4:4:end) = reshape(f22,size(DICmesh.coordinatesFEM,1),1);
        disp('--- Remove bad points done ---')
    % %%%%% Comment END %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    % ------ Plot ------
    USubpb1World = USubpb1; USubpb1World(2:2:end) = -USubpb1(2:2:end); 
    FSubpb1World = FSubpb1; FSubpb1World(2:4:end) = -FSubpb1World(2:4:end); FSubpb1World(3:4:end) = -FSubpb1World(3:4:end); 
    close all; Plotuv(USubpb1World,DICmesh.x0,DICmesh.y0World); 
    Plotdisp_show(USubpb1World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM,[],'NoEdgeColor');
    Plotstrain_show(FSubpb1World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM,[],'NoEdgeColor');
    save(['Subpb1_step',num2str(ALSolveStep)],'USubpb1','FSubpb1');
    fprintf('------------ Section 4 Done ------------ \n \n')
    
    
    %% Section 5: Subproblem 2 -- solve the global compatible displacement field
    fprintf('------------ Section 5 Start ------------ \n'); tic;
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is to solve the global step in ALDIC Subproblem 2
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % ======= ALStep 1 Subproblem 2: Global constraint =======
    % ------ Smooth displacements for a better F ------
    DICpara.DispFilterSize=0; DICpara.DispFilterStd=0; DICpara.StrainFilterSize=0; DICpara.StrainFilterStd=0; LevelNo=1;
    FSubpb1 = funSmoothStrain(FSubpb1,DICmesh,DICpara);
    
    % ====== Define penalty parameter ======
    mu = 1e-3; udual = 0*FSubpb1; vdual = 0*USubpb1;
    betaList = [1e-3,sqrt(1e-5),1e-2,sqrt(1e-3),1e-1,sqrt(1e-1)]*mean(DICpara.winstepsize).^2.*mu; % Tune beta in the betaList
    Err1 = zeros(length(betaList),1); Err2 = Err1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp(['***** Start step',num2str(ALSolveStep),' Subproblem2 *****'])
    % ====== Check to use FD or FE methods to solve Subpb2 step ======
    if DICpara.Subpb2FDOrFEM == 1 % Using FD method
        % ====== Build sparse finite difference operator ======
        disp('Assemble finite difference operator D');
        M = size(DICmesh.x0,1); N = size(DICmesh.x0,2);
        tic; Rad = 1; D = funDerivativeOp((M-2*Rad),(N-2*Rad),DICpara.winstepsize); % D = sparse(4*(M-2*Rad)*(N-2*Rad), 2*(M-2*Rad)*(N-2*Rad));
        D2 = funDerivativeOp(M,N,DICpara.winstepsize); toc
        disp('Finish assembling finite difference operator D');
        % ===== Solver using finite difference approximation ======
        tic; a = FSubpb1-udual; b = USubpb1-vdual;
        Rad = 1; [temp3,temp4] = funFDNeumannBCInd(size(DICmesh.coordinatesFEM,1),M,N,Rad); % Find coordinatesFEM that belong to (x(Rad+1:M-Rad,Rad+1:N-Rad),y(Rad+1:M-Rad,Rad+1:N-Rad))
        atemp = a(temp3); btemp = b(temp4); hbar = waitbar(0,'Please wait for Subproblem 2 global step!');
        for tempk = 1:length(betaList)
            beta = betaList(tempk);
            tempAMatrixSub2 = (beta*(D')*D) + mu*speye(2*(M-2*Rad)*(N-2*Rad));
            USubpb2temp = (tempAMatrixSub2) \ (beta*D'*atemp + mu*btemp ) ;
            USubpb2 = USubpb1; USubpb2(temp4) = USubpb2temp;
            FSubpb2 = D2*USubpb2;
            
            Err1(tempk) = norm(USubpb1-USubpb2,2);
            Err2(tempk) = norm(FSubpb1-FSubpb2,2);
            waitbar(tempk/(length(betaList)+1));
        end
        ErrSum = Err1+Err2*mean(DICpara.winstepsize)^2; [~,indexOfbeta] = min(ErrSum);
        
        try % Tune the best beta by a quadratic polynomial fitting
            [fitobj] = fit(log10(betaList(indexOfbeta-1:1:indexOfbeta+1))',ErrSum(indexOfbeta-1:1:indexOfbeta+1),'poly2');
            p = coeffvalues(fitobj); beta = 10^(-p(2)/2/p(1));
        catch, beta = betaList(indexOfbeta);
        end
        
        % Using the optimal beta to solve the ALDIC Subproblem 2 again
        tempAMatrixSub2 = (beta*(D')*D) + mu*speye(2*(M-2*Rad)*(N-2*Rad));
        USubpb2temp = (tempAMatrixSub2) \ (beta*D'*atemp + mu*btemp) ;
        USubpb2 = USubpb1; USubpb2(temp4) = USubpb2temp;
        waitbar(1); close(hbar);
        %%%%%%%%%%%%%% End of using finite difference approximation %%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    else %Subpb2FDOrFEM: Using FE method
        M = size(DICmesh.x0,1); N = size(DICmesh.x0,2); GaussPtOrder = 2; alpha = 0;
        close all; hbar = waitbar(0,'Please wait for Subproblem 2 global step!');
        % ====== Solver using finite element method ======
        for tempk = 1:length(betaList)
            beta = betaList(tempk);
            GaussPtOrder = 2; alpha = 0; [USubpb2] = Subpb2(DICmesh,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,GaussPtOrder);
            [FSubpb2] = funGlobal_NodalStrainAvg(DICmesh.coordinatesFEM,DICmesh.elementsFEM,USubpb2,GaussPtOrder);
            Err1(tempk) = norm(USubpb1-USubpb2,2);
            Err2(tempk) = norm(FSubpb1-FSubpb2,2);
            waitbar(tempk/(length(betaList)+1));
        end
        Err1Norm = (Err1-mean(Err1))/std(Err1); figure, plot(Err1Norm);
        Err2Norm = (Err2-mean(Err2))/std(Err2); figure, plot(Err2Norm);
        ErrSum = Err1Norm+Err2Norm; figure,plot(ErrSum); [~,indexOfbeta] = min(ErrSum);
        
        try % Tune the best beta by a quadratic polynomial fitting
            [fitobj] = fit(log10(betaList(indexOfbeta-1:1:indexOfbeta+1))',ErrSum(indexOfbeta-1:1:indexOfbeta+1),'poly2');
            p = coeffvalues(fitobj); beta = 10^(-p(2)/2/p(1));
        catch, beta = betaList(indexOfbeta);
        end
        
        % Using the optimal beta to solve the ALDIC Subproblem 2 again
        [USubpb2] = Subpb2(DICmesh,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,GaussPtOrder);
        USubpb2 = full(USubpb2);
        waitbar(1); close(hbar);
    end
    ALSub2Time(ALSolveStep) = toc; toc
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % ------- Before computing strain, we smooth the displacement field -------
    %USubpb2 = funSmoothDisp(USubpb2,coordinatesFEM,elementsFEM,x0,y0,winstepsize,DispFilterSize,DispFilterStd);
    % ------- Compute strain field --------
    if DICpara.Subpb2FDOrFEM == 1 %FD
        FSubpb2 = D2*USubpb2; % D2 = funDerivativeOp(M,N,winstepsize);
    else %FEM
        [FSubpb2] = funGlobal_NodalStrainAvg(DICmesh.coordinatesFEM,DICmesh.elementsFEM,USubpb2,GaussPtOrder);
    end
    
    % ------- Smooth strain field --------
    FSubpb2 = funSmoothStrain(FSubpb2,DICmesh,DICpara);
    
    % ------- Save data ------
    save(['Subpb2_step',num2str(ALSolveStep)],'USubpb2','FSubpb2');
    
    % ------ Plot ------
    USubpb2World = USubpb2; USubpb2World(2:2:end) = -USubpb2(2:2:end); 
    FSubpb2World = FSubpb2; FSubpb2World(2:4:end) = -FSubpb2World(2:4:end); FSubpb2World(3:4:end) = -FSubpb2World(3:4:end); 
    close all; Plotuv(USubpb2World,DICmesh.x0,DICmesh.y0World); 
    % Plotdisp_show(USubpb2World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM,[],'NoEdgeColor');
    % Plotstrain_show(FSubpb2World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM,[],'NoEdgeColor');
    
    % ======= Update dual variables =======
    if DICpara.Subpb2FDOrFEM == 1 %FD
        udualtemp1 = (FSubpb2 - FSubpb1); udualtemp2 = udualtemp1(temp3);
        vdualtemp1 = (USubpb2 - USubpb1); vdualtemp2 = vdualtemp1(temp4);
        udual = zeros(4*M*N,1); vdual = zeros(2*M*N,1);
        udual(temp3) = udualtemp2; vdual(temp4) = vdualtemp2;
    else  % FEM or other methods
        udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;
    end
    save(['uvdual_step',num2str(ALSolveStep)],'udual','vdual');
    fprintf('------------ Section 5 Done ------------ \n \n')
    
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Section 6: ADMM iterations
    fprintf('------------ Section 6 Start ------------ \n')
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This section is to run ADMM iteration: Subproblem 1 & 2
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % ==================== ADMM AL Loop ==========================
    ALSolveStep = 1; tol2 = 1e-4; UpdateY = 1e4; CrackOrNot = 0; CrackPath1 = [0,0]; CrackPath2 = [0,0]; CrackTip = [0,0];
    HPar = cell(21,1); for tempj = 1:21, HPar{tempj} = HtempPar(:,tempj); end
    
    while (ALSolveStep < 5)
        ALSolveStep = ALSolveStep + 1;  % Update using the last step
        %%%%%%%%%%%%%%%%%%%%%%% Subproblem 1 %%%%%%%%%%%%%%%%%%%%%%%%%
        disp(['***** Start step',num2str(ALSolveStep),' Subproblem1 *****']);
        tic;[USubpb1,~,ALSub1Timetemp,ConvItPerEletemp,LocalICGNBadPtNumtemp] = Subpb1( ...
                                        USubpb2,FSubpb2,udual,vdual,DICmesh.coordinatesFEM,...
                                        Df,fNormalized,gNormalized,mu,beta,HPar,ALSolveStep,DICpara,'GaussNewton',tol);
        FSubpb1 = FSubpb2; toc
        ALSub1Time(ALSolveStep) = ALSub1Timetemp; ConvItPerEle(:,ALSolveStep) = ConvItPerEletemp;  ALSub1BadPtNum(ALSolveStep) = LocalICGNBadPtNumtemp;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ------  Manually find some bad points from Local Subset ICGN step ------
        % disp('--- Start to manually remove bad points --- \n')
        % disp('    Comment codes here if you do not have bad local points \n')
        % %%%%% Comment START %%%%%
        %   close all; Plotuv(USubpb1,DICmesh.x0,DICmesh.y0);
        %   u = reshape(USubpb1(1:2:end),M,N); v = reshape(USubpb1(2:2:end),M,N);
        %   [u,v,~,Subpb1_BadptRow,Subpb1_BadptCol] = funRemoveOutliers(u',v',[],0.5,100,Local_BadptRow,Local_BadptCol); u=u';v=v';
        %   disp('--- Remove bad points done ---')
        %   USubpb1(1:2:end) = reshape(u,size(DICmesh.coordinatesFEM,1),1); 
        %   USubpb1(2:2:end) = reshape(v,size(DICmesh.coordinatesFEM,1),1);
        %   close all; Plotuv(USubpb1,DICmesh.x0,DICmesh.y0); Plotdisp_show(USubpb1,DICmesh.coordinatesFEM,DICmesh.elementsFEM);
        % %%%%% Comment END %%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        save(['Subpb1_step',num2str(ALSolveStep)],'USubpb1','FSubpb1');
        USubpb1 = funSmoothDisp(USubpb1,DICmesh,DICpara);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % ============== Subproblem 2 ==============
        disp(['***** Start step',num2str(ALSolveStep),' Subproblem2 *****'])
        if DICpara.Subpb2FDOrFEM == 1 %FD
            % ------- using finite difference approximation --------
            tic; a = FSubpb1-udual; b = USubpb1-vdual; atemp = a(temp3); btemp = b(temp4);
            USubpb2temp = (tempAMatrixSub2) \ (beta*D'*atemp + mu*btemp ) ;
            USubpb2 = USubpb1; USubpb2(temp4) = USubpb2temp; %toc
            % ------- End of using finite difference approximation --------
        else % FEM
            tic; [USubpb2] = Subpb2(DICmesh,beta,mu,USubpb1,FSubpb1,udual,vdual,alpha,GaussPtOrder);
            USubpb2 = full(USubpb2);
        end
        ALSub2Time(ALSolveStep) = toc; toc
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % ------- Before computing strain, we smooth the displacement field -------
        USubpb2 = funSmoothDisp(USubpb2,DICmesh,DICpara);
        % ------- Compute strain field --------
        if DICpara.Subpb2FDOrFEM == 1 %FD
            FSubpb2 = D2*USubpb2; % D2 = funDerivativeOp(M,N,winstepsize);
        else %FEM
            GaussPtOrder = 2; [FSubpb2] = funGlobal_NodalStrainAvg(DICmesh.coordinatesFEM,DICmesh.elementsFEM,USubpb2,GaussPtOrder);
        end
        % ------- Smooth strain field --------
        FSubpb2 = funSmoothStrain(FSubpb2,DICmesh,DICpara);
        
        save(['Subpb2_step',num2str(ALSolveStep)],'USubpb2','FSubpb2');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute norm of UpdateY
        USubpb2_Old = load(['Subpb2_step',num2str(ALSolveStep-1)],'USubpb2');
        USubpb2_New = load(['Subpb2_step',num2str(ALSolveStep)],'USubpb2');
        USubpb1_Old = load(['Subpb1_step',num2str(ALSolveStep-1)],'USubpb1');
        USubpb1_New = load(['Subpb1_step',num2str(ALSolveStep)],'USubpb1');
        if (mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit) ~= 0 && (ImgSeqNum>2)) || (ImgSeqNum < DICpara.ImgSeqIncUnit)
            UpdateY = norm((USubpb2_Old.USubpb2 - USubpb2_New.USubpb2), 2)/sqrt(size(USubpb2_Old.USubpb2,1));
            try
                UpdateY2 = norm((USubpb1_Old.USubpb1 - USubpb1_New.USubpb1), 2)/sqrt(size(USubpb1_Old.USubpb1,1));
            catch
            end
        end
        try
            disp(['Update local step  = ',num2str(UpdateY2)]);
            disp(['Update global step = ',num2str(UpdateY)]);
        catch
        end
        fprintf('*********************************** \n \n');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Update dual variables------------------------------
        if DICpara.Subpb2FDOrFEM == 1 %FD
            udualtemp1 =  (FSubpb2 - FSubpb1); udualtemp2 = udualtemp1(temp3);
            vdualtemp1 =  (USubpb2 - USubpb1); vdualtemp2 = vdualtemp1(temp4);
            udual(temp3) = udual(temp3)+udualtemp2;
            vdual(temp4) = vdual(temp4)+vdualtemp2;
        else %FEM
            udual = FSubpb2 - FSubpb1; vdual = USubpb2 - USubpb1;
        end
        
        save(['uvdual_step',num2str(ALSolveStep)],'udual','vdual');
        try
            if UpdateY < tol2 || UpdateY2 < tol2
                break
            end
        catch
        end
        
    end
    fprintf('------------ Section 6 Done ------------ \n \n')
    
    % Save data
    ResultDisp{ImgSeqNum-1}.U = full(USubpb2);
    ResultDisp{ImgSeqNum-1}.ALSub1BadPtNum = ALSub1BadPtNum;
    ResultDefGrad{ImgSeqNum-1}.F = full(FSubpb2); % tempFoamAL;
    
end

if DICpara.ImgSeqIncUnit < length(ImgNormalized) + 1
     fprintf('------------ Getting accumulated displacement field ------------ \n \n')
     Refcoodx=ResultFEMeshEachFrame{1, 1}.coordinatesFEM(:,1);
     Refcoody=ResultFEMeshEachFrame{1, 1}.coordinatesFEM(:,2);
     DefCoodx=ResultFEMeshEachFrame{1, 1}.coordinatesFEM(:,1);
     DefCoody=ResultFEMeshEachFrame{1, 1}.coordinatesFEM(:,2);
      for SeqNum = 1:length(ResultDisp)
             DefCood{SeqNum}.x= DefCoodx;
             DefCood{SeqNum}.y= DefCoody;
             ResultInc2AccDisp{SeqNum,1}= zeros(length(ResultDisp{1, 1}.U),1);
             xcoef=rbfcreate([ResultFEMeshEachFrame{SeqNum}.coordinatesFEM(:,1)';ResultFEMeshEachFrame{SeqNum}.coordinatesFEM(:,2)'], ResultDisp{SeqNum}.U(1:2:end)');
             ycoef=rbfcreate([ResultFEMeshEachFrame{SeqNum}.coordinatesFEM(:,1)';ResultFEMeshEachFrame{SeqNum}.coordinatesFEM(:,2)'], ResultDisp{SeqNum}.U(2:2:end)');
             xdef = rbfinterp([DefCoodx'; DefCoody'], xcoef);
             ydef = rbfinterp([DefCoodx'; DefCoody'], ycoef);
             ResultInc2AccDisp{SeqNum,1}(1:2:end)=DefCoodx-Refcoodx+xdef';
             ResultInc2AccDisp{SeqNum,1}(2:2:end)=DefCoody-Refcoody+ydef';
             if mod(SeqNum,DICpara.ImgSeqIncUnit) == 0 && SeqNum > 1 
                 DefCoodx=DefCoodx+xdef';
                 DefCoody=DefCoody+ydef';
             end
      end
end

% ------ Plot ------
USubpb2World = USubpb2; USubpb2World(2:2:end) = -USubpb2(2:2:end); 
FSubpb2World = FSubpb2; FSubpb2World(2:4:end) = -FSubpb2World(2:4:end); FSubpb2World(3:4:end) = -FSubpb2World(3:4:end); 
close all; Plotuv(USubpb2World,DICmesh.x0,DICmesh.y0World);
Plotdisp_show(USubpb2World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM);
Plotstrain_show(FSubpb2World,DICmesh.coordinatesFEMWorld,DICmesh.elementsFEM);

% ------ Save results ------
% Find img name and save all the results
[~,imgname,imgext] = fileparts(file_name{1,end});
results_name = ['results_',imgname,'_ws',num2str(DICpara.winsize),'_st',num2str(DICpara.winstepsize),'.mat'];
if DICpara.ImgSeqIncUnit < length(ImgNormalized)+1 
    save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame','ALSub1Time','ALSub2Time','ALSolveStep'...
        ,'Refcoodx','Refcoody','ResultInc2AccDisp');
else
    save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame','ALSub1Time','ALSub2Time','ALSolveStep');
end


%% Section 7: Check convergence
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is to check convergence of ADMM
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('------------ Section 7 Start ------------ \n')
% ====== Check convergence ======
fprintf('***** Check convergence ***** \n');
ALSolveStep1 = min(6,ALSolveStep);
disp('==== uhat^(k) - u^(k) ====');
for ALSolveStep = 1:ALSolveStep1
    USubpb2 = load(['Subpb2_step',num2str(ALSolveStep )],'USubpb2');
    USubpb1 = load(['Subpb1_step',num2str(ALSolveStep )],'USubpb1');
    UpdateY = norm((USubpb2.USubpb2 - USubpb1.USubpb1), 2)/sqrt(length(USubpb2.USubpb2));
    disp(num2str(UpdateY));
end
disp('==== Fhat^(k) - F^(k) ====');
for ALSolveStep = 1:ALSolveStep1
    FSubpb1 = load(['Subpb1_step',num2str(ALSolveStep )],'FSubpb1');
    FSubpb2 = load(['Subpb2_step',num2str(ALSolveStep )],'FSubpb2');
    UpdateF = norm((FSubpb1.FSubpb1 - FSubpb2.FSubpb2), 2)/sqrt(length(FSubpb1.FSubpb1));
    disp(num2str(UpdateF));
end
disp('==== uhat^(k) - uhat^(k-1) ====');
for ALSolveStep = 2:ALSolveStep1
    USubpb2_Old = load(['Subpb2_step',num2str(ALSolveStep-1)],'USubpb2');
    USubpb2_New = load(['Subpb2_step',num2str(ALSolveStep)],'USubpb2');
    UpdateY = norm((USubpb2_Old.USubpb2 - USubpb2_New.USubpb2), 2)/sqrt(length(USubpb2.USubpb2));
    disp(num2str(UpdateY));
end
disp('==== udual^(k) - udual^(k-1) ====');
for ALSolveStep = 2:ALSolveStep1
    uvdual_Old = load(['uvdual_step',num2str(ALSolveStep-1)],'udual');
    uvdual_New = load(['uvdual_step',num2str(ALSolveStep)],'udual');
    UpdateW = norm((uvdual_Old.udual - uvdual_New.udual), 2)/sqrt(length(uvdual_Old.udual));
    disp(num2str(UpdateW));
end
disp('==== vdual^(k) - vdual^(k-1) ====');
for ALSolveStep = 2:ALSolveStep1
    uvdual_Old = load(['uvdual_step',num2str(ALSolveStep-1)],'vdual');
    uvdual_New = load(['uvdual_step',num2str(ALSolveStep)],'vdual');
    Updatev = norm((uvdual_Old.vdual - uvdual_New.vdual), 2)/sqrt(length(uvdual_Old.vdual));
    disp(num2str(Updatev));
end
fprintf('------------ Section 7 Done ------------ \n \n')

% ------ Delete temp files ------
%%%%% Comment START %%%%%
% Uncomment these lines to delete temporary files
%   for tempi = 1:ALSolveStep
%       file_name_Subpb1 = ['Subpb1_step',num2str(tempi),'.mat'];
%       file_name_Subpb2 = ['Subpb2_step',num2str(tempi),'.mat'];
%       file_name_dual = ['uvdual_step',num2str(tempi),'.mat'];
%       delete(file_name_Subpb1); delete(file_name_Subpb2); delete(file_name_dual);
%   end
%%%%% Comment END %%%%%

% ------ clear temp variables ------
clear a ALSub1BadPtNum ALSub1Timetemp atemp b btemp cc ConvItPerEletemp hbar Hbar



%% Section 8: Compute strains
fprintf('------------ Section 8 Start ------------ \n')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is to compute strain fields and plot disp and strain results
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------ Convert units from pixels to the physical world ------
DICpara.um2px = funParaInput('ConvertUnit');
% ------ Smooth displacements ------
DICpara.DoYouWantToSmoothOnceMore = funParaInput('SmoothDispOrNot');
% ------ Choose strain computation method ------
DICpara.MethodToComputeStrain = funParaInput('StrainMethodOp');
% ------ Choose strain type (infinitesimal, Eulerian, Green-Lagrangian) ------
DICpara.StrainType = funParaInput('StrainType');
% ------ Choose image to plot (first only, second and next images) ------
if length(ImgNormalized)==2, DICpara.Image2PlotResults = funParaInput('Image2PlotResults');
else DICpara.Image2PlotResults = 1; % Plot over current, deformed image by default
end
% ------ Save fig format ------
DICpara.MethodToSaveFig = funParaInput('SaveFigFormat');
% ------ Choose overlay image transparency ------
DICpara.OrigDICImgTransparency = 1;
if DICpara.MethodToSaveFig == 1
    DICpara.OrigDICImgTransparency = funParaInput('OrigDICImgTransparency');
end

% ------ Start main part ------
for ImgSeqNum = 2 : length(ImgNormalized)
    
    disp(['Current image frame #: ', num2str(ImgSeqNum),'/',num2str(length(ImgNormalized))]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fNormalizedNewIndex = ImgSeqNum-mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit)-1;
    if DICpara.ImgSeqIncUnit ==  length(ImgNormalized)+1
        FEMeshIndLast = floor(fNormalizedNewIndex/DICpara.ImgSeqIncUnit);
    elseif DICpara.ImgSeqIncUnit >= 1
        FEMeshIndLast = floor(fNormalizedNewIndex/DICpara.ImgSeqIncUnit)-1;
    end
    FEMeshInd = FEMeshIndLast + 1;
    
    if FEMeshInd == 1
        USubpb2 = ResultDisp{ImgSeqNum-1}.U; %+ ResultDisp{10}.U + ResultDisp{20}.U;
        coordinatesFEM = ResultFEMesh{1}.coordinatesFEM;
        elementsFEM = ResultFEMesh{1}.elementsFEM;
        if (ImgSeqNum-1 == 1) || (DICpara.ImgSeqIncROIUpdateOrNot==1), UFEMesh = 0*USubpb2; end
    else
        USubpb2 = ResultDisp{ImgSeqNum-1}.U;
         UFEMesh = 0*USubpb2;
%         if mod(ImgSeqNum-2,DICpara.ImgSeqIncUnit) == 0
%             coordinatesFEM = ResultFEMesh{FEMeshInd}.coordinatesFEM;
%             elementsFEM = ResultFEMesh{FEMeshInd}.elementsFEM;
%             coordinatesFEMLast = ResultFEMesh{FEMeshIndLast}.coordinatesFEM;
%             UFEMeshLast = ResultDisp{ImgSeqNum-2}.U + UFEMesh;
%             xq = coordinatesFEM(:,1); yq = coordinatesFEM(:,2);
%             UFEMesh = 0*USubpb2;
%             UFEMesh(1:2:end) = griddata(coordinatesFEMLast(:,1),coordinatesFEMLast(:,2),UFEMeshLast(1:2:end),xq,yq,'v4');
%             UFEMesh(2:2:end) = griddata(coordinatesFEMLast(:,1),coordinatesFEMLast(:,2),UFEMeshLast(2:2:end),xq,yq,'v4');
%         end
% 
        if DICpara.ImgSeqIncUnit < (length(ImgNormalized)+1) && (ImgSeqNum-1) > DICpara.ImgSeqIncUnit
            UpdateRefIdx=0:DICpara.ImgSeqIncUnit:length(ResultDisp);
            UpdateRefIdxLast=max(UpdateRefIdx(find(ImgSeqNum-1 >UpdateRefIdx)));
            coordinatesFEM = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM;
            xq = coordinatesFEM(:,1); yq = coordinatesFEM(:,2);
%             xcoef=rbfcreate([DefCood{UpdateRefIdxLast}.x';DefCood{UpdateRefIdxLast}.y'],ResultInc2AccDisp{UpdateRefIdxLast}(1:2:end)');
%             ycoef=rbfcreate([DefCood{UpdateRefIdxLast}.x';DefCood{UpdateRefIdxLast}.y'],ResultInc2AccDisp{UpdateRefIdxLast}(2:2:end)');
%             UFEMesh(1:2:end) = rbfinterp([xq';yq'], xcoef);
%             UFEMesh(2:2:end) = rbfinterp([xq';yq'], ycoef);
            UFEMesh(1:2:end) = griddata(Refcoodx,Refcoody,ResultInc2AccDisp{ImgSeqNum-1}(1:2:end),xq,yq,'v4');
            UFEMesh(2:2:end) = griddata(Refcoodx,Refcoody,ResultInc2AccDisp{ImgSeqNum-1}(2:2:end),xq,yq,'v4');
            UFEMesh= UFEMesh- USubpb2;
        end
        USubpb2 = USubpb2 + UFEMesh;
    end
    
    FSubpb2 = ResultDefGrad{ImgSeqNum-1}.F;
    coordinatesFEM = ResultFEMeshEachFrame{ImgSeqNum-1}.coordinatesFEM;
    elementsFEM = ResultFEMeshEachFrame{ImgSeqNum-1}.elementsFEM;
    xList = min(coordinatesFEM(:,1)):DICpara.winstepsize:max(coordinatesFEM(:,1)); M = length(xList);
    yList = min(coordinatesFEM(:,2)):DICpara.winstepsize:max(coordinatesFEM(:,2)); N = length(yList);
    [x0,y0] = ndgrid(xList,yList);  
    x0 = x0-reshape(UFEMesh(1:2:end),size(x0,1),size(x0,2));  
    y0 = y0-reshape(UFEMesh(2:2:end),size(y0,1),size(y0,2));  
    x0World = DICpara.um2px*x0;
    y0World = DICpara.um2px*y0; % Ignore this: (size(ImgNormalized{1},2)+1-y0);
    coordinatesFEMWorld = DICpara.um2px*[coordinatesFEM(:,1),size(ImgNormalized{1},2)+1-coordinatesFEM(:,2)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     
    % ------ Plotting and Compute Strain-------
    if size(USubpb2,1) == 1
        ULocal = USubpb2_New.USubpb2; FLocal = FSubpb2.FSubpb2;
    else
        ULocal = USubpb2; FLocal = FSubpb2;
    end
    UWorld = DICpara.um2px*ULocal; UWorld(2:2:end) = -UWorld(2:2:end); % close all; Plotuv(UWorld,x0,y0World);
    
    % ------ Smooth displacements ------
    %prompt = 'Do you want to smooth displacement? (0-yes; 1-no)';
    %DoYouWantToSmoothOnceMore = input(prompt);
    SmoothTimes = 0;
    try
        while DICpara.DoYouWantToSmoothOnceMore == 0 && SmoothTimes < 3
            ULocal = funSmoothDisp(ULocal,DICmesh,DICpara);
            %%DICpara.DoYouWantToSmoothOnceMore = input(prompt);
            SmoothTimes = SmoothTimes + 1;
        end
    catch
    end
     
    % ----- Compute strain field ------
    ComputeStrain; % run ComputeStrain.m
    % %%%%% Add filter and plot strain field %%%%%
    % %%%%% Plotstrain_Fij; %%%%%
    
    % ------ Plot disp and strain ------
    close all; Plotuv(ULocal,x0,y0); 
    
    if DICpara.OrigDICImgTransparency == 1
        Plotdisp_show(UWorld,coordinatesFEMWorld,elementsFEM,DICpara);
        [strainxCoord,strainyCoord,dispu,dispv,dudx,dvdx,dudy,dvdy,strain_exx,strain_exy,strain_eyy,strain_principal_max, ...
            strain_principal_min,strain_maxshear,strain_vonMises]  =  Plotstrain0( ...
            UWorld,FStrainWorld,Rad,x0World,y0World,size(ImgNormalized{1}),DICpara);
    
    else % Plot over raw DIC images
        if DICpara.Image2PlotResults == 0 % Plot over the first image; "file_name{1,1}" corresponds to the first image
            Plotdisp(UWorld,x0World,y0World,size(ImgNormalized{1}),file_name{1,1},DICpara);
            [strainxCoord,strainyCoord,dispu,dispv,dudx,dvdx,dudy,dvdy,strain_exx,strain_exy,strain_eyy,strain_principal_max, ...
                strain_principal_min,strain_maxshear,strain_vonMises]  =  Plotstrain( ...
                UWorld,FStrainWorld,Rad,x0World,y0World,size(ImgNormalized{1}),file_name{1,1},DICpara);
        
        else % Plot over second or next deformed images
            Plotdisp(UWorld,x0World,y0World,size(ImgNormalized{1}),file_name{1,ImgSeqNum},DICpara);
            [strainxCoord,strainyCoord,dispu,dispv,dudx,dvdx,dudy,dvdy,strain_exx,strain_exy,strain_eyy,strain_principal_max, ...
                strain_principal_min,strain_maxshear,strain_vonMises]  =  Plotstrain( ...
                UWorld,FStrainWorld,Rad,x0World,y0World,size(ImgNormalized{1}),file_name{1,ImgSeqNum},DICpara);
        end
    end
    
    % ----- Save strain results ------
    ResultStrainWorld{ImgSeqNum-1} = struct('strainxCoord',strainxCoord,'strainyCoord',strainyCoord, ...
        'dispu',dispu,'dispv',dispv,'dudx',dudx,'dvdx',dvdx,'dudy',dudy,'dvdy',dvdy, ...
        'strain_exx',strain_exx,'strain_exy',strain_exy,'strain_eyy',strain_eyy, ...
        'strain_principal_max',strain_principal_max,'strain_principal_min',strain_principal_min, ...
        'strain_maxshear',strain_maxshear,'strain_vonMises',strain_vonMises);
    
    % ------ Save figures for tracked displacement and strain fields ------
    SaveFigFilesDispAndStrain;
    
    
end
% ------ END of for-loop {ImgSeqNum = 2:length(ImgNormalized)} ------
fprintf('------------ Section 8 Done ------------ \n \n')


% ------ Save data again including solved strain fields ------
results_name = ['results_',imgname,'_ws',num2str(DICpara.winsize),'_st',num2str(DICpara.winstepsize),'.mat'];
save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame',...
    'ALSub1Time','ALSub2Time','ALSolveStep','ResultStrainWorld');

if DICpara.ImgSeqIncUnit < length(ImgNormalized) 
     save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame',...
    'ALSub1Time','ALSub2Time','ALSolveStep','ResultStrainWorld','Refcoodx','Refcoody','ResultInc2AccDisp');
else
    save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame',...
    'ALSub1Time','ALSub2Time','ALSolveStep','ResultStrainWorld');
end


%% Section 9: Compute stress
fprintf('------------ Section 9 Start ------------ \n')
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This section is to compute stress fields and plot stress fields
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------ Choose material model ------
DICpara.MaterialModel = funParaInput('MaterialModel');
% ------ Define parameters in material models ------
if (DICpara.MaterialModel == 1) || (DICpara.MaterialModel == 2) % Linear elasticity
    fprintf('Define Linear elasticity parameters \n')
    fprintf("Young's modulus (unit: Pa): \n"); prompt = 'Input here (e.g., 69e9): '; 
    DICpara.MaterialModelPara.YoungsModulus = input(prompt); 
    fprintf("Poisson's ratio: \n"); prompt = 'Input here (e.g., 0.3): '; 
    DICpara.MaterialModelPara.PoissonsRatio = input(prompt);
    fprintf('------------------------------------- \n');
end

% ------ Start main part ------
for ImgSeqNum = 2 : length(ImgNormalized)
    
    disp(['Current image frame #: ', num2str(ImgSeqNum),'/',num2str(length(ImgNormalized))]); close all;
    
    % ------ Plot stress ------
    if DICpara.OrigDICImgTransparency == 1
        [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
            stress_principal_min_xyplane, stress_maxshear_xyplane, ...
            stress_maxshear_xyz3d, stress_vonMises]  =  Plotstress0( ...
            DICpara,ResultStrainWorld{ImgSeqNum-1},size(ImgNormalized{1}));
        
    else % Plot over raw DIC images
        if DICpara.Image2PlotResults == 0 % Plot over the first image; "file_name{1,1}" corresponds to the first image
            [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
                stress_principal_min_xyplane, stress_maxshear_xyplane, ...
                stress_maxshear_xyz3d, stress_vonMises] = Plotstress( ...
                DICpara,ResultStrainWorld{ImgSeqNum-1},size(ImgNormalized{1}),file_name{1,1});
            
        else % Plot over second or next deformed images
            [stress_sxx,stress_sxy,stress_syy, stress_principal_max_xyplane, ...
                stress_principal_min_xyplane, stress_maxshear_xyplane, ...
                stress_maxshear_xyz3d, stress_vonMises] = Plotstress( ...
                DICpara,ResultStrainWorld{ImgSeqNum-1},size(ImgNormalized{1}),file_name{1,ImgSeqNum});
            
        end
    end
    
    
    % ------ Save figures for computed stress fields ------
    SaveFigFilesStress;
    
    % ----- Save strain results ------
    ResultStressWorld{ImgSeqNum-1} = struct('stressxCoord',ResultStrainWorld{ImgSeqNum-1}.strainxCoord,'stressyCoord',ResultStrainWorld{ImgSeqNum-1}.strainyCoord, ...
        'stress_sxx',stress_sxx,'stress_sxy',stress_sxy,'stress_syy',stress_syy, ...
        'stress_principal_max_xyplane',stress_principal_max_xyplane, 'stress_principal_min_xyplane',stress_principal_min_xyplane, ...
        'stress_maxshear_xyplane',stress_maxshear_xyplane,'stress_maxshear_xyz3d',stress_maxshear_xyz3d, ...
        'stress_vonMises',stress_vonMises);
    
end
% ------ END of for-loop {ImgSeqNum = 2:length(ImgNormalized)} ------
fprintf('------------ Section 9 Done ------------ \n \n')

% ------ Save data again including solved stress fields ------
results_name = ['results_',imgname,'_ws',num2str(DICpara.winsize),'_st',num2str(DICpara.winstepsize),'.mat'];
save(results_name, 'file_name','DICpara','DICmesh','ResultDisp','ResultDefGrad','ResultFEMesh','ResultFEMeshEachFrame',...
    'ALSub1Time','ALSub2Time','ALSolveStep','ResultStrainWorld','ResultStressWorld');





