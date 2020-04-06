function [F_struc,K,KCut] = lmi2sedumistruct(F)
%lmi2sedumistruct   Internal function: Converts LMI to format needed in SeDuMi
% 
% % Author Johan L�fberg
% % $Id: lmi2sedumistruct.m,v 1.21 2006/09/22 08:18:37 joloef Exp $

nvars = yalmip('nvars'); %Needed lot'sa times...

% We first browse to see what we have got and the
% dimension of F_struc (massive speed improvement)
%top = 0;
type_of_constraint = zeros(size(F.clauses,2),1);
for i = 1:size(F.clauses,2)
    type_of_constraint(i) = F.clauses{i}.type;
%    [n,m] = size(F.clauses{i}.data);
%    [n,m] = size(getbasematrixwithoutcheck(F.clauses{i}.data,0));
%    top = top+n*m;
    % Is it a complex linear cone
 %   if (type_of_constraint(i)==2) & (~isreal(F.clauses{i}.data))
 %       top = top+n*m; % We will have constraints on Re and Im
 %   end
end

F_struc = [];

sdp_con = find(type_of_constraint == 1 | type_of_constraint == 9);
lin_con = find(type_of_constraint == 2 | type_of_constraint == 12);
equ_con = find(type_of_constraint == 3);
qdr_con = find(type_of_constraint == 4);
rlo_con = find(type_of_constraint == 5);

% SeDuMi struct
K.f = 0;
K.l = 0;
K.q = 0;
K.r = 0;
K.s = 0;
K.rank = [];
K.dualrank = [];
K.scomplex = [];
K.xcomplex = [];

KCut.f = [];
KCut.l = [];
KCut.q = [];
KCut.r = [];
KCut.s = [];

top = 1;
localtop = 1;
% Linear equality constraints
for i = 1:length(equ_con)
    constraints = equ_con(i);
    data = getbase(F.clauses{constraints}.data);
    [n,m] = size(F.clauses{constraints}.data);
    % Which variables are needed in this constraint
    lmi_variables = getvariables(F.clauses{constraints}.data);
    if isreal(data)
        ntimesm = n*m; %Just as well pre-calc			
    else
        % Complex constraint, Expand to real and Imag
        ntimesm = 2*n*m; %Just as well pre-calc
        data = [real(data);imag(data)];			
    end
    mapX = [1 1+lmi_variables];
    [ix,jx,sx] = find(data);
    F_structemp = sparse(ix,mapX(jx),sx,ntimesm,1+nvars);
    %F_structemp  = spalloc(ntimesm,1+nvars,0);
    %F_structemp(:,[1 1+lmi_variables(:)'])= data;

    % ...and add them together (efficient for large structures)
    %	F_struc(top:top+ntimesm-1,:) = F_structemp;	
    F_struc = [F_struc;F_structemp];
    
    if F.clauses{constraints}.cut
        KCut.f = [KCut.f localtop:localtop+ntimesm-1];
    end
    
    localtop = localtop+ntimesm;	    
    top = top+ntimesm;	
    K.f = K.f+ntimesm;
end

% Linear inequality constraints
localtop = 1;
F_struc = F_struc';
% [F_struc,K,KCut] = recursive_lp_fix(F,F_struc,K,KCut,lin_con,nvars,8,1);
% F_struc = F_struc';

% 
for i = 1:length(lin_con)
    constraints = lin_con(i);
    data = getbase(F.clauses{constraints}.data);
    [n,m] = size(F.clauses{constraints}.data);
    
    % Which variables are needed in this constraint
    lmi_variables = getvariables(F.clauses{constraints}.data);
    
    % Convert to real problem
    if isreal(data)
        ntimesm = n*m; %Just as well pre-calc			
    else
        % Complex constraint, Expand to real and Imag
        ntimesm = 2*n*m; %Just as well pre-calc
        data = [real(data);imag(data)];			
    end
    
    % Add numerical data to complete problem setup
    mapX = [1 1+lmi_variables];    
    [ix,jx,sx] = find(data);   
    F_structemp = sparse(mapX(jx),ix,sx,1+nvars,ntimesm);
    F_struc = [F_struc F_structemp];
    
    if F.clauses{constraints}.cut
        KCut.l = [KCut.l localtop:localtop+ntimesm-1];
    end

    localtop = localtop+ntimesm;	
    top = top+ntimesm;	
    K.l = K.l+ntimesm;
end
F_struc = F_struc';

[F_struc,K,KCut] = recursive_socp_fix(F,F_struc',K,KCut,qdr_con,nvars,8,1);
F_struc = F_struc';


% Rotated Lorentz cone constraints
for i = 1:length(rlo_con)
    constraints = rlo_con(i);
    
    [n,m] = size(F.clauses{constraints}.data);
    ntimesm = n*m; %Just as well pre-calc
    
    % Which variables are needed in this constraint
    lmi_variables = getvariables(F.clauses{constraints}.data);
    
    % We allocate the structure blockwise...
    F_structemp  = spalloc(ntimesm,1+nvars,0);
    % Add these rows only 
    F_structemp(:,[1 1+lmi_variables(:)'])= getbase(F.clauses{constraints}.data);
    % ...and add them together (efficient for large structures)
    %	F_struc(top:top+ntimesm-1,:) = F_structemp;
    F_struc = [F_struc;F_structemp];
    
    top = top+ntimesm;
    K.r(i) = n;	
end

% Semidefinite  constraints
% We append the recursively in order to speed up construction
% of problems with a lot of medium size SDPs
[F_struc,K,KCut] = recursive_sdp_fix(F,F_struc.',K,KCut,sdp_con,nvars,8,1);
F_struc = F_struc.';

% Now fix things for the rank constraint
% This is currently a hack...
% Should not be in this file
[rank_variables,dual_rank_variables] = yalmip('rankvariables');
if ~isempty(rank_variables)    
    used_in = find(sum(abs(F_struc(:,1+rank_variables)),2));
    if ~isempty(used_in)
        if used_in >=1+K.f & used_in < 1+K.l+K.f
            for i = 1:length(used_in)
                [ii,jj,kk] = find(F_struc(used_in(i),:));
                if length(ii)==2 & kk(2)<1
                    r = floor(kk(1));
                    var = jj(2)-1;
                    extstruct = yalmip('extstruct',var);
                    X = extstruct.arg{1};
                    if issymmetric(X)
                        F_structemp = sedumize(X,nvars);
                    else
                        error('Only symmetric matrices can be rank constrained.')
                    end
                    F_struc = [F_struc;F_structemp];
                    if isequal(K.s,0)
                        K.s(1,1) = size(extstruct.arg{1},1);
                    else
                        K.s(1,end+1) = size(extstruct.arg{1},1);
                    end
                    K.rank(1,end+1) = min(r,K.s(end));
                else
                    error('This rank constraint is not supported (only supports rank(X) < r)')
                end
            end
            % Remove the nonlinear operator constraints
            
            F_struc(used_in,:) = [];
            K.l = K.l - length(used_in);
        else
            error('You have added a rank constraint on an equality constraint, or a scalar expression?!')
        end
    end
end
if ~isempty(dual_rank_variables)
    used_in = find(sum(abs(F_struc(:,1+dual_rank_variables)),2));
    if ~isempty(used_in)
        if used_in >=1+K.f & used_in < 1+K.l+K.f
            for i = 1:length(used_in)
                [ii,jj,kk] = find(F_struc(used_in(i),:));
                if length(ii)==2 & kk(2)<1
                    r = floor(kk(1));
                    var = jj(2)-1;
                    extstruct = yalmip('extstruct',var);
                    X = extstruct.arg{1};
                    id = getlmiid(X);
                    inlist=getlmiid(F);
                    index=find(id==inlist);
                    if ~isempty(index)                                            
                        K.rank(1,index) = min(r,K.s(index));
                    end
                else
                    error('This rank constraint is not supported (only supports rank(X) < r)')
                end
            end
            % Remove the nonlinear operator constraints
            F_struc(used_in,:) = [];
            K.l = K.l - length(used_in);
        else
            error('You have added a rank constraint on an equality constraint, or a scalar expression?!')
        end
    end
end

function F_structemp = sedumize(Fi,nvars)
Fibase = getbase(Fi);
[n,m] = size(Fi);
ntimesm = n*m;
lmi_variables = getvariables(Fi);
[ix,jx,sx] = find(Fibase);
mapX = [1 1+lmi_variables];
F_structemp = sparse(ix,mapX(jx),sx,ntimesm,1+nvars);

function [F_struc,K,KCut] = recursive_lp_fix(F,F_struc,K,KCut,lp_con,nvars,maxnlp,startindex)

% Check if we should recurse
if length(lp_con)>2*maxnlp
    % recursing costs, so do 4 in one step
    ind = 1+ceil(length(lp_con)*(0:0.25:1));
    [F_struc1,K,KCut] = recursive_lp_fix(F,[],K,KCut,lp_con(ind(1):ind(2)-1),nvars,maxnlp,startindex+ind(1)-1);
    [F_struc2,K,KCut] = recursive_lp_fix(F,[],K,KCut,lp_con(ind(2):ind(3)-1),nvars,maxnlp,startindex+ind(2)-1);
    [F_struc3,K,KCut] = recursive_lp_fix(F,[],K,KCut,lp_con(ind(3):ind(4)-1),nvars,maxnlp,startindex+ind(3)-1);
    [F_struc4,K,KCut] = recursive_lp_fix(F,[],K,KCut,lp_con(ind(4):ind(5)-1),nvars,maxnlp,startindex+ind(4)-1);
    F_struc = [F_struc F_struc1 F_struc2 F_struc3 F_struc4];
    return
elseif length(lp_con)>maxnlp
    mid = ceil(length(lp_con)/2);
    [F_struc1,K,KCut] = recursive_lp_fix(F,[],K,KCut,lp_con(1:mid),nvars,maxnlp,startindex);
    [F_struc2,K,KCut] = recursive_lp_fix(F,[],K,KCut,lp_con(mid+1:end),nvars,maxnlp,startindex+mid);
    F_struc = [F_struc F_struc1 F_struc2];
    return
end

oldF_struc = F_struc;
F_struc = [];
for i = 1:length(lp_con)
    constraints = lp_con(i);
    Fi = F.clauses{constraints}.data;
    Fibase = getbase(Fi);
    [n,m] = size(Fi);
    
    %ntimesm = n*m; %Just as well pre-calc
    
    
    % Convert to real problem
    if isreal(Fibase)
        ntimesm = n*m; %Just as well pre-calc			
    else
        % Complex constraint, Expand to real and Imag
        ntimesm = 2*n*m; %Just as well pre-calc
        Fibase = [real(Fibase);imag(Fibase)];			
    end

    % Which variables are needed in this constraint
    lmi_variables = getvariables(Fi);
    mapX = [1 1+lmi_variables];
    
    [ix,jx,sx] = find(Fibase);
    
    F_structemp = sparse(mapX(jx),ix,sx,1+nvars,ntimesm);
    F_struc = [F_struc F_structemp];
       
    if F.clauses{constraints}.cut
        KCut.l = [KCut.l i+startindex-1];
    end
    
    K.l(i+startindex-1) = n;   
end
K.l = sum(K.l);
F_struc = [oldF_struc F_struc];


function [F_struc,K,KCut] = recursive_sdp_fix(F,F_struc,K,KCut,sdp_con,nvars,maxnsdp,startindex)

% Check if we should recurse
if length(sdp_con)>2*maxnsdp
    % recursing costs, so do 4 in one step
    ind = 1+ceil(length(sdp_con)*(0:0.25:1));
    [F_struc1,K,KCut] = recursive_sdp_fix(F,[],K,KCut,sdp_con(ind(1):ind(2)-1),nvars,maxnsdp,startindex+ind(1)-1);
    [F_struc2,K,KCut] = recursive_sdp_fix(F,[],K,KCut,sdp_con(ind(2):ind(3)-1),nvars,maxnsdp,startindex+ind(2)-1);
    [F_struc3,K,KCut] = recursive_sdp_fix(F,[],K,KCut,sdp_con(ind(3):ind(4)-1),nvars,maxnsdp,startindex+ind(3)-1);
    [F_struc4,K,KCut] = recursive_sdp_fix(F,[],K,KCut,sdp_con(ind(4):ind(5)-1),nvars,maxnsdp,startindex+ind(4)-1);
    F_struc = [F_struc F_struc1 F_struc2 F_struc3 F_struc4];
    return
elseif length(sdp_con)>maxnsdp
    mid = ceil(length(sdp_con)/2);
    [F_struc1,K,KCut] = recursive_sdp_fix(F,[],K,KCut,sdp_con(1:mid),nvars,maxnsdp,startindex);
    [F_struc2,K,KCut] = recursive_sdp_fix(F,[],K,KCut,sdp_con(mid+1:end),nvars,maxnsdp,startindex+mid);
    F_struc = [F_struc F_struc1 F_struc2];
    return
end

oldF_struc = F_struc;
F_struc = [];
for i = 1:length(sdp_con)
    constraints = sdp_con(i);
    Fi = F.clauses{constraints}.data;
    Fibase = getbase(Fi);
    [n,m] = size(Fi);
    ntimesm = n*m; %Just as well pre-calc

    % Which variables are needed in this constraint
    lmi_variables = getvariables(Fi);
    mapX = [1 1+lmi_variables];
    
    [ix,jx,sx] = find(Fibase);
    
    F_structemp = sparse(mapX(jx),ix,sx,1+nvars,ntimesm);
    F_struc = [F_struc F_structemp];

    if F.clauses{constraints}.cut
        KCut.s = [KCut.s i+startindex-1];
    end
    K.s(i+startindex-1) = n;
    K.rank(i+startindex-1) = n;
    K.dualrank(i+startindex-1) = n;
    % Check for a complex structure
    if ~isreal(F_structemp)
        K.scomplex = [K.scomplex i+startindex-1];
    end
end
F_struc = [oldF_struc F_struc];





function [F_struc,K,KCut] = recursive_socp_fix(F,F_struc,K,KCut,qdr_con,nvars,maxnsocp,startindex);

% Check if we should recurse
if length(qdr_con)>2*maxnsocp
    % recursing costs, so do 4 in one step
    ind = 1+ceil(length(qdr_con)*(0:0.25:1));
    [F_struc1,K,KCut] = recursive_socp_fix(F,[],K,KCut,qdr_con(ind(1):ind(2)-1),nvars,maxnsocp,startindex+ind(1)-1);
    [F_struc2,K,KCut] = recursive_socp_fix(F,[],K,KCut,qdr_con(ind(2):ind(3)-1),nvars,maxnsocp,startindex+ind(2)-1);
    [F_struc3,K,KCut] = recursive_socp_fix(F,[],K,KCut,qdr_con(ind(3):ind(4)-1),nvars,maxnsocp,startindex+ind(3)-1);
    [F_struc4,K,KCut] = recursive_socp_fix(F,[],K,KCut,qdr_con(ind(4):ind(5)-1),nvars,maxnsocp,startindex+ind(4)-1);
    F_struc = [F_struc F_struc1 F_struc2 F_struc3 F_struc4];
    return
elseif length(qdr_con)>maxnsocp
    mid = ceil(length(qdr_con)/2);
    [F_struc1,K,KCut] = recursive_socp_fix(F,[],K,KCut,qdr_con(1:mid),nvars,maxnsocp,startindex);
    [F_struc2,K,KCut] = recursive_socp_fix(F,[],K,KCut,qdr_con(mid+1:end),nvars,maxnsocp,startindex+mid);
    F_struc = [F_struc  F_struc1  F_struc2];
    return
end

% second order cone constraints
for i = 1:length(qdr_con)
    constraints = qdr_con(i);

    [n,m] = size(F.clauses{constraints}.data);
    ntimesm = n*m; %Just as well pre-calc

    % Which variables are needed in this constraint
    lmi_variables = getvariables(F.clauses{constraints}.data);

    data = getbase(F.clauses{constraints}.data);
    if isreal(data)
        mapX = [1 1+lmi_variables];
        [ix,jx,sx] = find(data);
        F_structemp = sparse(mapX(jx),ix,sx,1+nvars,ntimesm);
    else
        n = n+(n-1);
        ntimesm = n*m;
        F_structemp  = spalloc(ntimesm,1+nvars,0);
        data = [data(1,:);real(data(2:end,:));imag(data(2:end,:))];
        F_structemp(:,[1 1+lmi_variables(:)'])= data;
        F_structemp = F_structemp';
    end
    % ...and add them together (efficient for large structures)
    F_struc = [F_struc F_structemp];
    K.q(i+startindex-1) = n;
end






