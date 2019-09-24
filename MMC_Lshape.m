function [z]=MMC_Lshape(popposition,nf,it,nPop)
variables;
% set intial
ini_val=[0.29 0.04 0.06 0.04 0.7];
DW=2;DH=2;nelx=80;nely=80;x_int=0.4;y_int=0.4;volfrac=0.4;
% FEM data initialization
M = [ nely + 1 , nelx + 1 ];
EW = DW / nelx;  % length of element
EH = DH / nely;  % width of element
[ x ,y ] = meshgrid( EW * [ 0 :  nelx] , EH * [ 0 : nely]);
LSgrid.x = x(:);
LSgrid.y = y(:);     % coordinate of nodes
% Material properties
h=1;  %thickness
E=1;
nu=0.3;
% Component geometry initialization
x0=x_int/2:x_int:DW;    % x-coordinates of the centers of components
y0=y_int/2:y_int:DH;    % y-coordinates of the centers of components
xn=length(x0);               % number of component groups in x direction               
yn=length(y0);               % number of component groups in y direction
x0=kron(x0,ones(1,2*yn));                                         
y0=repmat(kron(y0,ones(1,2)),1,xn);   
N=length(x0);                % total number of components in the design domain
L=repmat(ini_val(1),1,N);                    % vector of the half length of each component
t1=repmat(ini_val(2),1,N);                   % vector of the half width of component at point A
t2=repmat(ini_val(3),1,N);                   % vector of the half width of component at point B
t3=repmat(ini_val(4),1,N);                   % vector of the half width of component at point C
st=repmat([ini_val(5) -ini_val(5)],1,N/2);   % vector of the sine value of the inclined angle of each component 
variable=[x0;y0;L;t1;t2;t3;st];
%Parameter of MMA 
xy00=variable(:);
xval=xy00;
xold1 = xy00;
xold2 = xy00;
%Limits of variable:[x0 ; y0  ; L ; t1 ; t2; t3; st]; 
xmin=[0 ; 0  ; 0.01 ; 0.01  ; 0.01 ; 0.03; -1.0];
xmin=repmat(xmin,N,1);
xmax=[DW ; DH ; 2.0 ; 0.2 ; 0.2 ; 0.2 ; 1.0];
xmax=repmat(xmax,N,1);
low   = xmin;
upp   = xmax;
m = 1;  %number of constraint
Var_num=7;  % number of design variables for each component
nn=Var_num*N;
c=1000*ones(m,1);
d=zeros(m,1);
a0=1;
a=zeros(m,1);
%Define loads and supports(Short beam)
fixeddofs   = [2*(nely+1)-1:2*(nely+1):2*(nely+1)*(2*nelx/5+1)-1,2*(nely+1):2*(nely+1):2*(nely+1)*(2*nelx/5+1)];
alldofs     = 1:2*(nely+1)*(nelx+1);
freedofs    = setdiff(alldofs,fixeddofs);
loaddof=((nely+1)*nelx+nely/5+1)*2;
F=sparse(loaddof,1,-1,2*(nely+1)*(nelx+1),1);
%Preparation FE analysis
nodenrs=reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);        
edofVec=reshape(2*nodenrs(1:end-1,1:end-1)-1,nelx*nely,1);   
edofMat=repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 4 5] 2 3],nelx*nely,1);  
iK=kron(edofMat,ones(8,1))';
jK=kron(edofMat,ones(1,8))';
EleNodesID=edofMat(:,2:2:8)./2;    
iEner=EleNodesID';
[KE] = BasicKe(E,nu, EW, EH,h);   %  stiffness matrix k^s is formed
%Initialize iteration
p=6;                 
alpha=1e-3;            % parameter alpha in the Heaviside function
epsilon=4*min(EW,EH);  % regularization parameter epsilon in the Heaviside function
Phi=cell(N,1);        
Loop=1;
change=1;
maxiter=150;      % the maximum number of iterations
while change>0.001 && Loop<maxiter
    %forming void domain
    void=tvoid(LSgrid.x,LSgrid.y);
    %Forming Phi^s
    for ii=1:N
        Phi{ii}=tPhi(xy00(Var_num*ii-Var_num+1:Var_num*ii),LSgrid.x,LSgrid.y,p); 
    end
   %Union of components
    tempPhi_max=Phi{1};
    for ii=2:N
        tempPhi_max=max(tempPhi_max,Phi{ii});
    end
    final=min(tempPhi_max,void);
    final_max=reshape(final,nely+1,nelx+1);
%     %Plot components
%     contourf(reshape(x , M), reshape(y , M),final_max,[0,0]);
%     axis equal;axis([0 DW 0 DH]);pause(1e-6);
    % Calculating the finite difference quotient of H
    H=Heaviside(final_max,alpha,nelx,nely,epsilon);
    diffH=cell(N,1);
    for j=1:N
        for ii=1:Var_num
            xy001=xy00;
            xy001(ii+(j-1)*Var_num)=xy00(ii+(j-1)*Var_num)+max(2*min(EW,EH),0.005);
            tmpPhiD1=tPhi(xy001(Var_num*j-Var_num+1:Var_num*j),LSgrid.x,LSgrid.y,p);
            tempPhi_max1=tmpPhiD1;
            for ik=1:j-1
                tempPhi_max1=max(tempPhi_max1,Phi{ik});
            end
            for ik=j+1:N
                tempPhi_max1=max(tempPhi_max1,Phi{ik});
            end
             tempPhi_max01=min(tempPhi_max1,void);
            xy002=xy00;
            xy002(ii+(j-1)*Var_num)=xy00(ii+(j-1)*Var_num)-max(2*min(EW,EH),0.005);
            tmpPhiD2=tPhi(xy002(Var_num*j-Var_num+1:Var_num*j),LSgrid.x,LSgrid.y,p);
            tempPhi_max2=tmpPhiD2;
            for ik=1:j-1
                tempPhi_max2=max(tempPhi_max2,Phi{ik});
            end
            for ik=j+1:N
                tempPhi_max2=max(tempPhi_max2,Phi{ik});
            end
            tempPhi_max02=min(tempPhi_max2,void);
            HD1=Heaviside( tempPhi_max01,alpha,nelx,nely,epsilon);
            HD2=Heaviside(tempPhi_max02,alpha,nelx,nely,epsilon);
            diffH{j}(:,ii)=(HD1-HD2)/(2*(max(2*min(EW,EH),0.005)));
        end
    end
    %FEA
    denk = sum( H(EleNodesID).^2, 2 ) / 4; 
    den=sum( H(EleNodesID), 2 ) / 4;
    A1=sum(den)*EW*EH;                     
    U = zeros(2*(nely+1)*(nelx+1),1);
    sK = KE(:)*denk(:)';
    K = sparse(iK(:),jK(:),sK(:)); K = (K+K')/2;
    U(freedofs,:) = K(freedofs,freedofs)\F(freedofs,:);
    %Energy of element
    energy = sum((U(edofMat)*KE).*U(edofMat),2);
    sEner=ones(4,1)*energy'/4;
    energy_nod=sparse(iEner(:),1,sEner(:));   
    Comp=F'*U;
    % Sensitivities
    df0dx=zeros(Var_num*N,1);
    dfdx=zeros(Var_num*N,1);
    for k=1:N
            df0dx(Var_num*k-Var_num+1:Var_num*k,1)=2*energy_nod'.*H*diffH{k}; 
            n=sum(diffH{k})/4;
            dfdx(Var_num*k-Var_num+1:Var_num*k,1)=sum(diffH{k})/4;
    end  
    %MMA optimization
    f0val =Comp;     
    df0dx=-df0dx/max(abs(df0dx));
    fval=A1/(DW*DH-1)-volfrac;          
    dfdx=dfdx/max(abs(dfdx));    
    %     [xmma,ymma,zmma,lam,xsi,eta,mu,zet,ss,low,upp] = ...
    %         mmasub1(m,nn,Loop,xval,xmin,xmax,xold1,xold2, ...
    %         f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d);
    df0dx2=0;
    dfdx2=0;
    [xmma,ymma,zmma,lam,xsi,eta,mu,zet,ss,low,upp] = ...
    mmasub(m,nn,Loop,xval,xmin,xmax,xold1,xold2, ...
    f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,popposition);
    xold2 = xold1;
    xold1 = xval;
    change=max(abs(xval-xmma));
    xval = xmma;
    xy00=round(xval*1e4)/1e4;
    disp([' It.: ' sprintf('%4i\t',Loop) ' Obj.: ' sprintf('%6.3f\t',f0val) ' Vol.: ' ...
        sprintf('%6.4f\t',fval) 'ch.:' sprintf('%6.4f\t',change)]);
    Loop = Loop + 1;
end
z=f0val;
path_in='D:\超级桌面\DG_MMC 副本\L\';     % path_in为保存路径，根据需要修改?
figure;
contourf(reshape(x , M), reshape(y , M),final_max,[0,0]);
axis equal;
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
% axis([0 DW 0 DH]);
pause(1e-6);
if it==1
   ini_it_pso=ini_it_pso+1;
end
if ini_it_pso<=nPop
    name=strcat(num2str(it-1),'0',num2str(nf));
else
    name=strcat(num2str(it),'0',num2str(nf));
end
saveas(gca,[path_in,name],'jpg');    % 保存图片（以数字命名）
close;
end

%Forming Phi_i for each component

%Heaviside function

%Element stiffness matrix

%~~~~ A Moving Morphable Components (MMC) based topology optimization code
%~~~~ by Xu Guo Weisheng Zhang and Jie Yuan 
%~~~~ Department of Engineering Mechanics, State Key Laboratory of Structural Analysis       
%~~~~ for Industrial Equipment, Dalian University of Technology
%~~~~ Please send your suggestions and comments to guoxu@dlut.edu.cn