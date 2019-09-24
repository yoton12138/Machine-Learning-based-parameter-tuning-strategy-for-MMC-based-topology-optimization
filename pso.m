%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPEA102
% Project Title: Implementation of Particle Swarm Optimization in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%
% 
clc;
clear;
close all;
variables;

%% Problem Definition
addpath(genpath('D:\超级桌面\DG_MMC 副本'));
MaxIt=1000;      % Maximum Number of Iterations MaxIt=10000

nPop=10;        % Population Size (Swarm Size) nPop=50
% main_initial_stastic;

CostFunction=@(x,i,it,nPop) MMC_Lshape(x,i,it,nPop);        % Cost Function

nVar = 4;    % Number of Decision Variables
VarSize = [1 nVar];

VarMin = [0.25 0.25 1.0 0.25];   % Lower Bound of Variables VarMin   befa init incr decr
VarMax = [0.75 0.75 1.5 0.75];   % Upper Bound of Variables VarMax   0.5  0.5  1.2  0.5

%% PSO Parameters
% PSO Parameters
w=1;            % Inertia Weight
wdamp=0.99;     % Inertia Weight Damping Ratio
c1=1.0;         % Personal Learning Coefficient
c2=2.0;         % Global Learning Coefficient

% Velocity Limits
VelMax=0.2*(VarMax-VarMin);
VelMin=-VelMax;

%% Initialization
% empty_particle.Position=[];
% empty_particle.Cost=[];
% empty_particle.Velocity=[];
% empty_particle.Best.Position=[];
% empty_particle.Best.Cost=[];
% particle=repmat(empty_particle,nPop,1);
empty_particle.Position=cell(1,MaxIt);
empty_particle.Cost=cell(1,MaxIt);
empty_particle.Velocity=cell(1,MaxIt);
empty_particle.Best.Position=cell(1,MaxIt);
empty_particle.Best.Cost=cell(1,MaxIt);
particle=repmat(empty_particle,nPop,1);
GlobalBest.Cost=inf;
a=1
while a
    it=1;
    ini_it_pso=0;
    disp('==================initial==================')
    for i=1:nPop
        % Initialize Position
        for j=1:4
            x(j)=unifrnd(VarMin(j),VarMax(j),1);
            x(j)=roundn(x(j),-4);
        end
        particle(i,:).Position{it}=x;
        % Initialize Velocity
        v = zeros(VarSize);

        particle(i).Velocity{it}=v;
        % Evaluation     
        particle(i).Cost{it}=CostFunction(particle(i).Position{it},i,it,nPop);

    end
    %% 调用一次极端随机森林
    system('python test.py');
    [Data]=importdata('class_results-ET-l.txt');
    data=Data.data;
    
    for i=1:nPop
        % Update Personal Best
        particle(i).Best.Position{it}=particle(i).Position{it};
        particle(i).Best.Cost{it}=particle(i).Cost{it};

        % Update Global Best
        if particle(i).Best.Cost{it}<GlobalBest.Cost && data(i,2)==1

            GlobalBest.Position=particle(i).Best.Position{it};
            GlobalBest.Cost=particle(i).Best.Cost{it};
            a=0
        end
    end  
end
system('python switch.py');%jpg2png
outputiterop=zeros(MaxIt,2);
BestCost=zeros(MaxIt,1);
%PSO Main Loop

for it=1:MaxIt
    
    for i=1:nPop
        
        % Update Velocity
        particle(i).Velocity{it+1} = w*particle(i).Velocity{it} ...
            +c1*rand(VarSize).*(particle(i).Best.Position{it}-particle(i).Position{it}) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position{it});
        
        % Apply Velocity Limits
        for j=1:4
        particle(i).Velocity{it+1}(j) = max(particle(i).Velocity{it+1}(j),VelMin(j));
        particle(i).Velocity{it+1}(j) = min(particle(i).Velocity{it+1}(j),VelMax(j));
        end
        % Update Position
        particle(i).Position{it+1} = particle(i).Position{it} + particle(i).Velocity{it+1};
        particle(i).Position{it+1} = roundn(particle(i).Position{it+1},-4);%圆整4位小数
        % Velocity Mirror Effect
        IsOutside=zeros(1,4);
        for j=1:4
        IsOutside(j)=(particle(i).Position{it+1}(j)<VarMin(j) | particle(i).Position{it+1}(j)>VarMax(j));
        end
        IsOutside=logical(IsOutside);
        particle(i).Velocity{it+1}(IsOutside)=-particle(i).Velocity{it+1}(IsOutside);
        % Apply Position Limits
        for j=1:4
        particle(i).Position{it+1}(j) = max(particle(i).Position{it+1}(j),VarMin(j));
        particle(i).Position{it+1}(j) = min(particle(i).Position{it+1}(j),VarMax(j));
        end
        % Evaluation
        particle(i).Cost{it+1} = CostFunction(particle(i).Position{it+1},i,it,nPop);
        
       %% 调用一次极端随机森林
        system('python test.py');
        [Data]=importdata('class_results-ET-l.txt');
        data=Data.data;
        system('python switch.py');
        % Update Personal Best
        if particle(i).Cost{it+1}<particle(i).Best.Cost{it} && data(1,2)==1
            %disp(['label:' sprintf('%4i\t',data(1,2))])
            particle(i).Best.Position{it+1}=particle(i).Position{it+1};
            particle(i).Best.Cost{it+1}=particle(i).Cost{it+1};
            
            % Update Global Best
            if particle(i).Best.Cost{it+1}<GlobalBest.Cost && data(1,2)==1
                GlobalBest.Position=particle(i).Best.Position{it+1};
                GlobalBest.Cost=particle(i).Best.Cost{it+1};
            end
        else
            particle(i).Best.Position{it+1}=particle(i).Best.Position{it};
            particle(i).Best.Cost{it+1}=particle(i).Best.Cost{it};
            GlobalBest.Position=GlobalBest.Position;
            GlobalBest.Cost=GlobalBest.Cost;
        end
        
    end
    BestCost(it)=GlobalBest.Cost;
    %output the optimum result of each iteration
    for ii=1:nPop
    if  particle(ii).Best.Cost{it+1}==BestCost(it)
        itt=ii;
        outputiterop(it,:)=[ii,it];
        break;
     end
    end
    
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
    w=w*wdamp;
    %收敛判断
    if it>2&&(BestCost(it)~=BestCost(it-1))
        cha = BestCost(it-1)-BestCost(it)
        if cha <0.005
            save all
            break;
        end
    end
    if it>50
        if BestCost(it) == BestCost(it-49)
            save all
            break;
        end
    end    
    save all
    disp('data saved')
end
BestSol = GlobalBest.Position;
% nPopff=find(particle.Best.Cost{MaxIt+1}==min(particle.Best.Cost{MaxIt+1}));
%% Results
path_in1='D:\超级桌面\MMC-SMO-改\';
figure;
plot(BestCost,'LineWidth',2);
% semilogy(BestCost,'LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
grid on;
name=strcat('pso','MMC');
saveas(gca,[path_in1,name],'jpg');    % 保存图片（以数字命名）
for ii=1:nPop
    if min(particle(ii).Best.Cost{it+1})==BestCost(MaxIt,1)
        itt=ii;
        break;
    end
end