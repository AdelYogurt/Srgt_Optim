matCL2CD = zeros(size(DataStruct,2),4);
vecNumMorph = zeros(size(DataStruct,2),1);
matMorph = [];
matMaAoAH = [];
matCl = [];
matCd = [];
matSref = [];
for i = 1:size(DataStruct,2)
    CLCD = DataStruct(i).cl./DataStruct(i).cd;
    cd
    [a,b] = max(CLCD);
    [c,d] = min(CLCD);
    matCL2CD(i,:) = [a,b,c,d];
%     num = size(DataStruct(i).morph,1);
    vecNumMorph(i) = size(DataStruct(i).morph,1);
    Index = randperm(vecNumMorph(i),15);
    matMorph = [ matMorph; DataStruct(i).morph(Index,:)];
    matMaAoAH = [matMaAoAH;  repmat([DataStruct(i).Ma, DataStruct(i).AoA, DataStruct(i).H],15,1)];
    matCl = [ matCl; DataStruct(i).cl(Index,:)];
    matCd = [ matCd; DataStruct(i).cd(Index,:)];
    matSref = [matSref; DataStruct(i).sref(Index,:)];
end

X  = [matMaAoAH,matMorph];

X(:,1) = ( X(:,1) - 7 )/10.5;
X(:,2) = ( X(:,2) - 10 )/10;
X(:,3) = ( X(:,3) - 30 )/30;
X(:,4) = ( X(:,4) )/(pi/6);
X(:,5) = X(:,5);
X(:,6) = X(:,6);

nv = size(X,2);

[IdxFolds] = srgtsGetKfolds(X, 20, 'MaxMin');
Index = reshape(IdxFolds,size(IdxFolds,1)/20,20);

R2  = zeros(20,1);
RMSE = zeros(20,1);
MAE = zeros(20,1);
for i = 1 :20
    DeletNum  = Index(:,i);
%     if i == 1
%         DeletNum  = (1: sum(vecNumMorph(1)))';
%     else
%         DeletNum  = (sum(vecNumMorph(1:i-1))+1 :1:   sum(vecNumMorph(1:i-1))+vecNumMorph(1:i))';
%     end
    matConX = X;
    matConYCL = matCl;
    matConYCD = matCd;
    matConYSref = matSref;
    
    matTestX = matConX(DeletNum,:);
%     Index1 = matTestX(:,1)>0.9;
%     Index2 = matTestX(:,3)<0.1;
%     Index = logical(Index1+Index2);
    matTestY = matSref(DeletNum,:);
%     matTestY(Index) =[];
%     matTestX(Index,:) =[];
    
    matConX(DeletNum,:) = [];
    matConYCL(DeletNum,:) = [];
    matConYCD(DeletNum,:) = [];
    matConYSref(DeletNum,:) = [];
    TrainX  = matConX;
    TrainY  = matConYSref;
    TestX   = matTestX;
    
    [inputn, inputps] = mapminmax(TrainX');
    [outputn, outputps] = mapminmax(TrainY');
    inputn_test = mapminmax('apply', TestX', inputps);
    
    % Define the parameter of BP training process
    NodeNum = 24; % Key Para :
    fprintf(num2str(NodeNum)); 
    TypeNum = size(TrainY,2);
    Epochs = 3000;
    
    % 'logsig' 'purelin' 'tansig'
    TF1 = 'logsig'; TF2 = 'purelin';
    
    % 'trainlm' 'trainrp'
    net = newff(minmax(inputn), [NodeNum TypeNum], {TF1, TF2}, 'trainscg');
    net.trainParam.epochs = Epochs;
    net.trainParam.goal = 1e-8;
    net.trainParam.min_grad = 1e-20;
    net.trainParam.show = 200;
    net.trainParam.time = inf;
%     net.trainParam.mu_max = 1e10;
    
    net.performFcn = 'mse';
    
    [net, tr, Y, E, Pf, Af] = train(net, inputn, outputn);
    
    pre_test_Y_BP   = sim(net, inputn_test);
    pre_test_Y_BP   = mapminmax('reverse', pre_test_Y_BP, outputps);
    FPred           = pre_test_Y_BP';

    R2(i) = 1 - sum((matTestY - FPred).^2 ) / sum((matTestY - mean(matTestY)).^2 );
    RMSE(i)  = sqrt( sum((matTestY-FPred).^2)/length(matTestY) );
    MAE(i)  = sum(abs(matTestY-FPred))/length(matTestY);
end
disp(mean(RMSE));
disp(mean(MAE));
% Rate = ( matCL2CD(:,1) - matCL2CD(:,3) ) ./ matCL2CD(:,3) ;
% 
% % X = [matMaAoAH,matMorph];
% X = matMorph;
% Y = matSref;
% 
% % X(:,1) = ( X(:,1)-7 )/10.5;
% % X(:,2) = ( X(:,2)-10)/10;
% % X(:,3) = ( X(:,3)-30)/30;
% X(:,1) = ( X(:,1)   )/(pi/6);
% % X(:,6) = ( X(:,6)- 0.2)/(2-0.2);
% vecNumDeletTest = randperm(length(Y),16)';
% X(vecNumDeletTest,:)=[];
% 
% [IdxFolds] = srgtsGetKfolds(X, 20, 'MaxMin');
% Index = 1:1:length(Y);
% 
% vecNumTest = randperm(length(Y),110)';
% vecNumCon  = (1:1:length(Y))';
% vecNumCon(vecNumTest) = [];
% 
% matConX  = X(vecNumCon,:);
% matConY  = Y(vecNumCon,:);
% 
% matTestX  = X(vecNumTest,:);
% matTestY  = Y(vecNumTest,:);
% 
% nv        = size(matTestX,2);
% c         = (  (  max(max(matConX)) -  min(min(matConX))  )/size(matConX,1)  )^(1/nv);
% RBFOption = srgtsRBFSetOptions(matConX, matConY, @rbf_build ,'IMQ' , c , 0 );
% RBFModel  = srgtsRBFFit(RBFOption);
% % KrgOption = srgtsKRGSetOptions(matConX, matConY, @dace_fit, @dace_regpoly0 , @dace_corrgauss, VecTheta0, eps*ones(1,nv), 20*ones(1,nv));
% % KrgModel  = srgtsKRGFit(KrgOption);
% 
% [matPredY] = srgtsRBFEvaluate(matTestX, RBFModel);
% 
% R2 = 1 - sum((matTestY - matPredY).^2 ) / sum((matTestY - mean(matTestY)).^2 );
% 
% nv        = size(X,2);
% c         = (  (  max(max(X)) -  min(min(X))  )/size(X,1)  )^(1/nv);
% SrefRBFOption = srgtsRBFSetOptions(X, Y, @rbf_build ,'IMQ' , c , 0 );
% SrefRBFModel  = srgtsRBFFit(SrefRBFOption);




