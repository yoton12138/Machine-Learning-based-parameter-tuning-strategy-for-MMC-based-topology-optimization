%处理数据
D = load('all.mat');
% zeros = [10,1];
pcost = [];%每个粒子的目标函数柔度值
pbest = [];%粒子最优参数
for i=1:10
    particle_cost = D.particle(i).Cost;
    particle_position = D.particle(i).Position;
    pcost = [pcost;particle_cost];
    pbest = [pbest;particle_position];
    i = i+1;
end
pcost(cellfun(@isempty,pcost))=[];
pbest(cellfun(@isempty,pbest))=[];

pcost = reshape(cell2mat(pcost),[10,112]);%看情况改列值
pbest = reshape(pbest,[10,112]);

[min_pcost,index]=min(pcost,[],2);%找最小柔度（最优解）及其位置

row = [1,2,3,4,5,6,7,8,9,10];
column = [];
for i=1:10
    column = [column,index(i)];
    i=i+1;
end
pbest = pbest(row+(column-1)*size(pbest,1));%按索引得出每个粒子的最优参数组合
pbest = cell2mat(pbest);
pbest = (reshape(pbest,[4,10]))';
pbest = [pbest,min_pcost];%参数和柔度对应合并

%将迭代过程的最优解提取出来
gbindex = D.outputiterop;
gbest=[];
bcost = D.BestCost;
bcost(find(bcost==0))=[];%去除零元素
iteration = size(bcost,1);
for i=1:iteration
    if i == 1
        j = gbindex(1,1);
        gbest = [gbest,D.particle(j).Position(i+1)];
    end
    if i~=1
        if D.BestCost(i) ~= D.BestCost(i-1)
            j = gbindex(i,1);
            gbest = [gbest,D.particle(j).Position(i+1)];
        end
        if D.BestCost(i) == D.BestCost(i-1)
           gbest = [gbest,gbest(i-1)];
        end
    end 
end
