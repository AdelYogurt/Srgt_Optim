clc;
clear;
close all hidden;

data_library_name = 'optimal_data_library';

benchmark = BenchmarkFunction();

% variable_number = 2;
% object_function = @(x) benchmark.singleGPObject(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = [-2,-2];
% up_bou = [2,2];
% nonlcon_function = [];
% cheapcon_function = [];

% variable_number = 2;
% object_function = @(x) benchmark.singlePKObject(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = [-3,-3];
% up_bou = [3,3];
% nonlcon_function = [];
% cheapcon_function = [];

% variable_number = 4;
% object_function = @(x) benchmark.singleROSObject(x);
% object_function_LF = @(x) benchmark.singleROSObjectLow(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = [-2,-2,-2,-2];
% up_bou = [2,2,2,2];
% nonlcon_function = [];
% nonlcon_function_LF = [];
% cheapcon_function = [];

% variable_number = 6;
% object_function = @(x) benchmark.singleHNObject(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = zeros(1,variable_number);
% up_bou = ones(1,variable_number);
% nonlcon_function = [];
% cheapcon_function = [];
% model_function = [];

% variable_number = 20;
% object_function = @(x) benchmark.singleDP20Object(x);
% object_function_LF = @(x) benchmark.singleDP20ObjectLow(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = ones(1,variable_number)*-30;
% up_bou = ones(1,variable_number)*30;
% nonlcon_function = [];
% nonlcon_function_LF = [];
% cheapcon_function = [];

variable_number = 20;
object_function = @(x) benchmark.singleEP20Object(x);
object_function_LF = @(x) benchmark.singleEP20ObjectLow(x);
A = [];
B = [];
Aeq = [];
Beq = [];
low_bou = ones(1,variable_number)*-30;
up_bou = ones(1,variable_number)*30;
nonlcon_function = [];
nonlcon_function_LF = [];
cheapcon_function = [];

% variable_number = 30;
% object_function = @(x) benchmark.singleAckley30Object(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = -15*ones(1,variable_number);
% up_bou = 20*ones(1,variable_number);
% nonlcon_function = [];
% cheapcon_function = [];

% variable_number = 2;
% object_function = @(x) benchmark.singleG06Object(x);
% object_function_LF = @(x) benchmark.singleG06ObjectLow(x);
% A = [];
% B = [];
% Aeq = [];
% Beq = [];
% low_bou = [13,0];
% up_bou = [100,100];
% nonlcon_function = @(x) benchmark.singleG06Nonlcon(x);
% nonlcon_function_LF = @(x) benchmark.singleG06NonlconLow(x);
% cheapcon_function = [];
% model_function = [];

% variable_number = 4;
% object_function = @(x) benchmark.singlePVD4Object(x);
% object_function_low = @(x) benchmark.singlePVD4ObjecttLow(x);
% A = [-1,0,0.0193,0;
%     0,-1,0.00954,0;];
% B = [0;0];
% Aeq = [];
% Beq = [];
% low_bou = [0,0,0,0];
% up_bou = [1,1,50,240];
% nonlcon_function = @(x) cheapconFunction(x,A,B,Aeq,Beq,@(x) benchmark.singlePVD4Nonlcon(x));
% cheapcon_function = [];
% model_function = [];

% variable_number = 13;
% object_function = @benchmark.singleG01Object;
% object_function_low = @benchmark.singleG01ObjectLow;
% A = [ 2   2   0   0   0   0   0   0   0   1   1   0   0;
%     2   0   2   0   0   0   0   0   0   1   0   1   0;
%     0   2   2   0   0   0   0   0   0   0   1   1   0;
%     -8  0   0   0   0   0   0   0   0   1   0   0   0;
%     0   -8  0   0   0   0   0   0   0   0   1   0   0;
%     0   0   0   -2  -1  0   0   0   0   1   0   0   0;
%     0   0   0   0   0   -2  -1  0   0   0   1   0   0;
%     0   0   0   0   0   0   0   -2  -1  0   0   1   0;
%     ];
% B = [10;10;10;0;0;0;0;0];
% Aeq = [];
% Beq = [];
% low_bou = zeros(1,13);
% up_bou = ones(1,13);
% up_bou(10:12) = 100;
% nonlcon_function = @(x) cheapconFunction(x,A,B,Aeq,Beq,[]);
% nonlcon_function_LF = @(x) cheapconFunction(x,A,B,Aeq,Beq,[]);
% cheapcon_function = [];

% x_initial = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
% fmincon_option = optimoptions('fmincon','Algorithm','sqp');
% [x_best,fval_best,~,output] = fmincon(object_function,x_initial,A,B,Aeq,Beq,low_bou,up_bou,nonlcon_function,fmincon_option)

%% single run

delete([data_library_name,'.txt']);
delete('result_total.txt');

[x_best,fval_best,NFE,output] = optimalSurrogateDYCORSLMSRBF...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,[],200,500)

result_x_best = output.result_x_best;
result_fval_best = output.result_fval_best;

figure(1);
plot(result_fval_best);

figure(2);
[x_list,fval_list,con_list,coneq_list] = dataLibraryLoad...
    (data_library_name,low_bou,up_bou);
scatter3(x_list(:,1),x_list(:,2),fval_list);
xlabel('X');
ylabel('Y');
zlabel('Z');

%% repeat run

% repeat_number = 10;
% result_fval = zeros(repeat_number,1);
% max_NFE = 200;
% for repeat_index = 1:repeat_number
%     delete([data_library_name,'.txt']);
%     delete('result_total.txt');
% 
%     [x_best,fval_best,NFE,output] = optimalSurrogateDYCORSLMSRBF...
%         (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
%         cheapcon_function,[],max_NFE);
% 
%     result_fval(repeat_index) = fval_best;
% end
% 
% fprintf('Fval     : lowest = %4.4f,mean = %4.4f,worst = %4.4f,std = %4.4f \n',min(result_fval),mean(result_fval),max(result_fval),std(result_fval));
% object_function_name = char(object_function);
% save([object_function_name(15:end-3),'_',num2str(max_NFE),'_DYCORS_LMSRBF','.mat']);

%% main
function [x_best,fval_best,NFE,output] = optimalSurrogateDYCORSLMSRBF...
    (object_function,variable_number,low_bou,up_bou,nonlcon_function,...
    cheapcon_function,model_function,....
    NFE_max,iteration_max,torlance,nonlcon_torlance)
% surrogate base optimal use radias base function method version 0
% all function exchange x should be colume vector
% x_list is x_number x variable_number matrix
% both nonlcon_function and cheapcon_function format is [con,coneq]
% model_function should output fval,format is [fval,con,coneq]
% con or coneq can be colume vector if there was more than one constrain
%
% referance: [1] 
%
% Copyright Adel 2022.10
%
if nargin < 11 || isempty(nonlcon_torlance)
    nonlcon_torlance = 1e-3;
    if nargin < 10 || isempty(torlance)
        torlance = 1e-3;
        if nargin < 9
            iteration_max = [];
            if nargin < 8
                NFE_max = [];
            end
        end
    end
end

if nargin < 7
    model_function = [];
    if nargin < 6
        cheapcon_function = [];
        if nargin < 5
            nonlcon_function = [];
        end
    end
end

DRAW_FIGURE_FLAG = 0; % whether draw data
INFORMATION_FLAG = 1; % whether print data
CONVERGENCE_JUDGMENT_FLAG = 0; % whether judgment convergence

if isempty(iteration_max)
    iteration_max = 100;
end

% hyper parameter
sample_number_initial = ceil(variable_number/10)*20;
trial_number = min(100*variable_number,100);
coord_select_prob_initial = min(20/variable_number,1);
sigma_coord_initial = 0.2*(up_bou-low_bou);
sigma_coord_min = 0.2*1/64*(up_bou-low_bou);
sigma_coord_max = 2*(up_bou-low_bou);
tau_success = 3;
tau_fail = max(variable_number,5);
kappa = 4;
w_list = [0.3,0.5,0.8,0.95];

% max fval when normalize fval,con,coneq
nomlz_fval = 10;

protect_range = 1e-4;

data_library_name = 'optimal_data_library';
file_result = fopen('result_total.txt','a');
fprintf(file_result,'%s\n',datetime);
fclose(file_result);
clear('file_result');

done = 0;NFE = 0;iteration = 0;

result_x_best = zeros(iteration_max,variable_number);
result_fval_best = zeros(iteration_max,1);

% if do not input model_function,generate model_function
if isempty(model_function)
    model_function = @(x) modelFunction(x,object_function,nonlcon_function);
end

iteration = iteration+1;

% step 2
% use latin hypercube method to get initial sample x_list
% [~,x_updata_list,~] = getLatinHypercube...
%     (sample_number_initial,variable_number,[],low_bou,up_bou,cheapcon_function);
x_updata_list = lhsdesign(sample_number_initial,variable_number,'iterations',100,'criterion','maximin').*(up_bou-low_bou)+low_bou;

% detech expensive constraints
if ~isempty(x_updata_list)
    [~,con,coneq] = dataLibraryUpdata...
        (data_library_name,model_function,x_updata_list(1,:));NFE = NFE+1;
    x_updata_list = x_updata_list(2:end,:);
else
    [~,con,coneq] = dataLibraryLoad(data_library_name,low_bou,up_bou);
end
if ~isempty(con) || ~isempty(coneq)
    expensive_nonlcon_flag = 1;
else
    expensive_nonlcon_flag = 0;
end

% NFE and iteration setting
if isempty(NFE_max)
    if expensive_nonlcon_flag
        NFE_max = 50*variable_number;
    else
        NFE_max = 20*variable_number;
    end
end

if isempty(iteration_max)
    if expensive_nonlcon_flag
        iteration_max = 50*variable_number;
    else
        iteration_max = 20*variable_number;
    end
end

% import data from data library
[x_list,fval_list,con_list,coneq_list] = dataLibraryLoad...
    (data_library_name,low_bou,up_bou);

% step 3
% updata data library by x_list
[fval_updata_list,con_updata_list,coneq_updata_list] = dataLibraryUpdata...
    (data_library_name,model_function,x_updata_list);NFE = NFE+size(x_updata_list,1);
x_list = [x_list;x_updata_list];
fval_list = [fval_list;fval_updata_list];
if ~isempty(con_updata_list)
    con_list = [con_list;con_updata_list];
end
if ~isempty(coneq_updata_list)
    coneq_list = [coneq_list;coneq_updata_list];
end

% nomalization all data
fval_max = max(abs(fval_list),[],1);
fval_nomlz_list = fval_list./fval_max*nomlz_fval;
if ~isempty(con_list)
    con_max_list = max(abs(con_list),[],1);
    con_nomlz_list = con_list./con_max_list*nomlz_fval;
else
    con_max_list = [];
    con_nomlz_list = [];
end
if ~isempty(coneq_list)
    coneq_max_list = max(abs(coneq_list),[],1);
    coneq_nomlz_list = coneq_list./coneq_max_list*nomlz_fval;
else
    coneq_max_list = [];
    coneq_nomlz_list = [];
end

% step 0
% find best result to record
[x_best,fval_best,con_best,coneq_best] = findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance);

result_x_best(iteration,:) = x_best;
result_fval_best(iteration,:) = fval_best;
iteration = iteration+1;

sigma_coord = sigma_coord_initial;
C_success = 0;
C_fail = 0;
while ~done
    % step 4
    % generate RBF model use normalization fval
    [radialbasis_model_fval,radialbasis_model_con,radialbasis_model_coneq,output_radialbasis] = getRadialBasisModel...
        (x_list,fval_nomlz_list,con_nomlz_list,coneq_nomlz_list);
    object_function_surrogate = output_radialbasis.object_function_surrogate;
    nonlcon_function_surrogate = output_radialbasis.nonlcon_function_surrogate;

    % calculate gradient
    [~,gradient] = differ(object_function_surrogate,x_best,[],1e-3);
    gradient = abs(gradient);
    gradient_max = max(gradient); gradient_min = min(gradient);
    gradient = (gradient-gradient_min)./(gradient_max-gradient_min);

    % step 5
    % perturbed to get trial point
    coord_select_prob = coord_select_prob_initial*(1-log(NFE-sample_number_initial+1)/log(NFE_max-sample_number_initial));

    % select coordinates to perturb
    coordinate_select_list = [];
    for variable_index = 1:variable_number
        if rand() < coord_select_prob*gradient(variable_index)
            coordinate_select_list = [coordinate_select_list,variable_index];
        end
    end
    if isempty(coordinate_select_list)
        coordinate_select_list = randi(variable_number);
    end

    % generate trial point
    x_trial_list = repmat(x_best,trial_number,1);
    for coordinate_select_index = 1:length(coordinate_select_list)
        coordinate_select = coordinate_select_list(coordinate_select_index);
        x_trial_list(:,coordinate_select) = x_trial_list(:,coordinate_select)+...
            normrnd(0,sigma_coord(coordinate_select),[trial_number,1]);
    end
    x_trial_list = max(x_trial_list,low_bou);
    x_trial_list = min(x_trial_list,up_bou);

    % step 6
    % select point to add
    w_index = mod(NFE-sample_number_initial+1,kappa);
    if w_index == 0
        w_R = w_list(kappa);
    else
        w_R = w_list(w_index);
    end
    w_D = 1-w_R;

    % evaluate trial point merit
    merit_list = meritFunction...
        (x_trial_list,object_function_surrogate,x_list,variable_number,...
        w_R,w_D);
    [~,index] = min(merit_list);
    x_potential = x_trial_list(index,:);

    % check x_potential if exist in data library
    % if not,updata data libraray
    [x_potential,fval_potential,con_potential,coneq_potential,NFE_p,repeat_index] = dataLibraryUpdataProtect...
        (data_library_name,model_function,x_potential,x_list,low_bou,up_bou,protect_range);NFE = NFE+NFE_p;
    x_list = [x_list;x_potential];
    fval_list = [fval_list;fval_potential];
    if ~isempty(con_list)
        con_list = [con_list;con_potential];
    end
    if ~isempty(coneq_list)
        coneq_list = [coneq_list;coneq_potential];
    end

    % normalization fval updata
    if ~isempty(fval_potential)
        fval_potential_nomlz = fval_potential/fval_max*nomlz_fval;
        fval_nomlz_list = [fval_nomlz_list;fval_potential_nomlz];
    end
    if ~isempty(con_potential)
        con_potential_nomlz = (con_potential./con_max_list)*nomlz_fval;
        con_nomlz_list = [con_nomlz_list;con_potential_nomlz];
    end
    if ~isempty(coneq_potential)
        coneq_potential_nomlz = (coneq_potential./coneq_max_list)*nomlz_fval;
        coneq_nomlz_list = [coneq_nomlz_list;coneq_potential_nomlz];
    end

    % when x_potential is exist in data library,x_potential_add will be
    % empty,this times we will use origin point data
    if isempty(x_potential)
        x_potential = x_list(repeat_index,:);
        fval_potential = fval_list(repeat_index,:);
        if ~isempty(con_list)
            con_potential = con_list(repeat_index,:);
        else
            con_potential = [];
        end
        if ~isempty(coneq_list)
            coneq_potential = coneq_list(repeat_index,:);
        else
            coneq_potential = [];
        end
    end

    % step 7
    % adjust step size
    if fval_potential < fval_best
        C_success = C_success+1;
        C_fail = 0;
    else
        C_success = 0;
        C_fail = C_fail+1;
    end

    if C_success >= tau_success
        sigma_coord = min(2*sigma_coord,sigma_coord_max);
        C_success = 0;
    end

    if C_fail >= tau_fail
        sigma_coord = max(sigma_coord/2,sigma_coord_min);
        C_fail = 0;
    end

    if INFORMATION_FLAG
        fprintf('iteration:          %-3d    NFE:    %-3d\n',iteration,NFE);
        fprintf('current x:          %s\n',num2str(x_potential));
        fprintf('current value:      %f\n',fval_potential);
        fprintf('current violation:  %s  %s\n',num2str(con_potential),num2str(coneq_potential));
        fprintf('\n');
    end

    if DRAW_FIGURE_FLAG && variable_number < 3
        interpVisualize(radialbasis_model_fval,low_bou,up_bou);
        line(x_potential(1),x_potential(2),fval_potential/fval_max*nomlz_fval,'Marker','o','color','r','LineStyle','none')
        hold on;
        scatter3(x_trial_list(:,1),x_trial_list(:,2),merit_list*10,merit_list*100,merit_list*10,'filled');
        hold off;
    end

    % forced interrupt
    if iteration > iteration_max || NFE >= NFE_max
        done = 1;
    end

    % convergence judgment
    if CONVERGENCE_JUDGMENT_FLAG
        if (iteration > 2 && ...
                abs((fval_potential-fval_potential_old)/fval_potential_old) < torlance)
            done = 1;
            if ~isempty(con_best)
                if sum(con_best > nonlcon_torlance)
                    done = 0;
                end
            end
            if ~isempty(coneq_best)
                if sum(abs(coneq_best) > nonlcon_torlance)
                    done = 0;
                end
            end
        end
    end

    % find best result to record
    [x_best,fval_best,con_best,coneq_best] = findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance);

    result_x_best(iteration,:) = x_best;
    result_fval_best(iteration,:) = fval_best;
    iteration = iteration+1;

end
result_x_best = result_x_best(1:iteration-1,:);
result_fval_best = result_fval_best(1:iteration-1);

output.result_x_best = result_x_best;
output.result_fval_best = result_fval_best;

    function [fval,con,coneq] = modelFunction...
            (x,object_function,nonlcon_function)
        % model function,concertrate fval,con,coneq into one function
        %
        if nargin < 3 || isempty(nonlcon_function)
            con = [];
            coneq = [];
        else
            [con,coneq] = nonlcon_function(x);
        end
        fval = object_function(x);
    end

    function [x_updata_list,fval_updata_list,con_updata_list,coneq_updata_list,NFE_updata,repeat_index] = dataLibraryUpdataProtect...
            (data_library_name,model_function,x_add_list,...
            x_list,low_bou,up_bou,protect_range)
        % function updata data with same_point_avoid protect
        % return fval
        % all list is x_number x variable_number matrix
        % notice if x_add is exist in library,point will be delete
        %
        variable_number__ = size(x_list,2);
        NFE_updata = 0;
        x_updata_list = [];fval_updata_list = [];con_updata_list = [];coneq_updata_list = [];repeat_index = [];
        for x_index__ = 1:size(x_add_list,1)
            x_updata__ = x_add_list(x_index__,:);

            % check x_potential if exist in data library
            % if not,updata data libraray
            distance__ = sum((abs(x_updata__-x_list)./(up_bou-low_bou)),2);
            [~,min_index__] = min(distance__);
            if distance__(min_index__) < variable_number__*protect_range
                % distance to exist point of point to add is small than protect_range
                repeat_index = [repeat_index;min_index__];
            else
                [fval_updata__,con_updata__,coneq_updata__] = dataLibraryUpdata...
                    (data_library_name,model_function,x_updata__);NFE_updata = NFE_updata+1;
                x_updata_list = [x_updata_list;x_updata__];
                fval_updata_list = [fval_updata_list;fval_updata__];
                con_updata_list = [con_updata_list;con_updata__];
                coneq_updata_list = [coneq_updata_list;coneq_updata__];
            end
        end
    end

    function fval = meritFunction...
            (x_trial_list,object_function_surrogate,x_list,variable_number,...
            w_R,w_D)
        % function to evaluate sample point
        %

        % value scale
        R = object_function_surrogate(x_trial_list);
        R_min = min(R); R_max = max(R);
        R = (R-R_min)./(R_max-R_min);

        % distance scale
        dis = zeros(size(x_trial_list,1),size(x_list,1));
        for vari_index = 1:variable_number
            dis = dis+(x_trial_list(:,vari_index)-x_list(:,vari_index)').^2;
        end
        D = min(sqrt(dis),[],2);
        D_min = min(D); D_max = max(D);
        D = (D_max-D)./(D_max-D_min);

        fval = w_R*R+w_D*D;
    end
end

%% auxiliary function
function [x_best,fval_best,exitflag,output] = findMinMSP...
    (object_function_surrogate,variable_number,low_bou,up_bou,nonlcon_function_surrogate,...
    cheapcon_function,nonlcon_torlance)
% find min fval use MSP guideline
% MSP: object_funtion is object_function (generate by surrogate model)
% nonlcon_function generate by surrogate model
% use ga as optimal method
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        nonlcon_function_surrogate = [];
        if nargin < 4
            up_bou = [];
            if nargin < 3
                low_bou = [];
            end
        end
    end
end
% % object function convert to penalty function if input nonlcon function
% if ~isempty(nonlcon_function)
%     object_function = @(x) penaltyFunction(object_function,x,nonlcon_function);
%     constraint_function = cheapcon_function;
% end

% obtian total constraint function
if ~isempty(nonlcon_function_surrogate) || ~isempty(cheapcon_function)
    constraint_function = @(x) totalconFunction...
        (x,nonlcon_function_surrogate,cheapcon_function,nonlcon_torlance);
else
    constraint_function = [];
end

% generate initial population for ga
population_matrix = zeros(max(10,2*variable_number),variable_number);
for population_index = 1:size(population_matrix,1)
    x = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
    if ~isempty(cheapcon_function)
        [con,coneq] = cheapcon_function(x);
        while sum([~(con < 0);abs(coneq) < 0])
            x = rand(1,variable_number).*(up_bou-low_bou)+low_bou;
            [con,coneq] = cheapcon_function(x);
        end
    end
    population_matrix(population_index,:) = x;
end

% optiaml
ga_option = optimoptions('ga','FunctionTolerance',1e-2,'ConstraintTolerance',1e-2,...
    'PopulationSize',max(10,2*variable_number),...
    'MaxGenerations',100,'InitialPopulationMatrix',population_matrix,...
    'display','none');
[x_best,fval_best,exitflag,output] = ga...
    (object_function_surrogate,variable_number,[],[],[],[],low_bou',up_bou',constraint_function,ga_option);
fmincon_option = optimoptions('fmincon','FunctionTolerance',1e-6,'ConstraintTolerance',1e-6,...
    'algorithm','sqp',....
    'display','none');
[x_best,fval_best,exitflag,output] = fmincon...
    (object_function_surrogate,x_best,[],[],[],[],low_bou,up_bou,constraint_function,fmincon_option);

    function [con,coneq] = totalconFunction...
            (x,nonlcon_function,cheapcon_function,nonlcon_torlance)
        con = [];
        coneq = [];
        if ~isempty(nonlcon_function)
            [expencon,expenconeq] = nonlcon_function(x);
            con = [con;expencon-nonlcon_torlance];
            coneq = [coneq;expenconeq-nonlcon_torlance];
        end
        if ~isempty(cheapcon_function)
            [expencon,expenconeq] = cheapcon_function(x);
            con = [con;expencon];
            coneq = [coneq;expenconeq];
        end
    end
end

function [x_best,fval_best,con_best,coneq_best] = findMinRaw...
    (x_list,fval_list,con_list,coneq_list,...
    cheapcon_function,nonlcon_torlance)
% find min fval in raw data
% x_list,rank is variable
% con_list,rank is con
% coneq_list,rank is coneq
% function will find min fval in con==0
% if there was not feasible x,find min consum
%
con_best = [];
coneq_best = [];
max_nonlcon_list = zeros(size(x_list,1),1);
max_cheapcon_list = zeros(size(x_list,1),1);
% process expendsive con
if ~isempty(con_list)
    max_nonlcon_list = max(con_list,[],2);
end
if ~isempty(coneq_list)
    max_nonlcon_list = max(abs(coneq_list),[],2);
end

% add cheap con
for x_index = 1:size(x_list,1)
    if ~isempty(cheapcon_function)
        [con,coneq] = cheapcon_function(x_list(x_index,:));
        max_cheapcon_list(x_index) = max_cheapcon_list(x_index)+...
            sum(max(con,0))+sum(coneq.*coneq);
    end
end

con_judge_list = (max_nonlcon_list > nonlcon_torlance)+...
    (max_cheapcon_list > 0);
index = find(con_judge_list == 0);
if ~isempty(index)
    % feasible x
    x_list = x_list(index,:);
    fval_list = fval_list(index);
    if ~isempty(con_list)
        con_list = con_list(index,:);
    end
    if ~isempty(coneq_list)
        coneq_list = coneq_list(index,:);
    end

    % min fval
    [fval_best,index_best] = min(fval_list);
    x_best = x_list(index_best,:);
    if ~isempty(con_list)
        con_best = con_list(index_best,:);
    end
    if ~isempty(coneq_list)
        coneq_best = coneq_list(index_best,:);
    end
else
    % min consum
    [~,index_best] = min(max_nonlcon_list);
    fval_best = fval_list(index_best);
    x_best = x_list(index_best,:);
    if ~isempty(con_list)
        con_best = con_list(index_best,:);
    end
    if ~isempty(coneq_list)
        coneq_best = coneq_list(index_best,:);
    end
end
end

%% surrogate model
function [radialbasis_model_fval,radialbasis_model_con,radialbasis_model_coneq,output] = getRadialBasisModel...
    (x_list,fval_list,con_list,coneq_list)
% base on library_data to create radialbasis model and function
% if input model,function will updata model
% object_function is single fval output
% nonlcon_function is normal nonlcon_function which include con,coneq
% con is colume vector,coneq is colume vector
% var_function is same
%
basis_function = @(r) r.^3;

[predict_function_fval,radialbasis_model_fval] = interpRadialBasisPreModel...
    (x_list,fval_list,basis_function);

if ~isempty(con_list)
    predict_function_con = cell(size(con_list,2),1);
    radialbasis_model_con = struct('X',[],'Y',[],...
        'radialbasis_matrix',[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_con = repmat(radialbasis_model_con,[size(con_list,2),1]);
    for con_index = 1:size(con_list,2)
        [predict_function_con{con_index},radialbasis_model_con(con_index)] = interpRadialBasisPreModel...
            (x_list,con_list(:,con_index));
    end
else
    predict_function_con = [];
    radialbasis_model_con = [];
end

if ~isempty(coneq_list)
    predict_function_coneq = cell(size(coneq_list,2),1);
    radialbasis_model_coneq = struct('X',[],'Y',[],...
        'radialbasis_matrix',[],[],'beta',[],...
        'aver_X',[],'stdD_X',[],'aver_Y',[],'stdD_Y',[],'basis_function',[],...
        'predict_function',[]);
    radialbasis_model_coneq = repmat(radialbasis_model_coneq,[size(coneq_list,2),1]);
    for coneq_index = 1:size(coneq_list,2)
        [predict_function_coneq{coneq_index},radialbasis_model_con(coneq_index)] = interpRadialBasisPreModel...
            (x_list,coneq_list(:,coneq_index));
    end
else
    predict_function_coneq = [];
    radialbasis_model_coneq = [];
end

object_function_surrogate = @(X_predict) objectFunctionSurrogate(X_predict,predict_function_fval);
if isempty(radialbasis_model_con) && isempty(radialbasis_model_coneq)
    nonlcon_function_surrogate = [];
else
    nonlcon_function_surrogate = @(X_predict) nonlconFunctionSurrogate(X_predict,predict_function_con,predict_function_coneq);
end

output.object_function_surrogate = object_function_surrogate;
output.nonlcon_function_surrogate = nonlcon_function_surrogate;

    function fval = objectFunctionSurrogate...
            (X_predict,predict_function_fval)
        % connect all predict favl
        %
        fval = predict_function_fval(X_predict);
    end
    function [con,coneq] = nonlconFunctionSurrogate...
            (X_predict,predict_function_con,predict_function_coneq)
        % connect all predict con and coneq
        %
        if isempty(predict_function_con)
            con = [];
        else
            con = zeros(size(X_predict,1),length(predict_function_con));
            for con_index__ = 1:length(predict_function_con)
                con(:,con_index__) = ....
                    predict_function_con{con_index__}(X_predict);
            end
        end
        if isempty(predict_function_coneq)
            coneq = [];
        else
            coneq = zeros(size(X_predict,1),length(predict_function_coneq));
            for coneq_index__ = 1:length(predict_function_coneq)
                coneq(:,coneq_index__) = ...
                    predict_function_coneq{coneq_index__}(X_predict);
            end
        end
    end
end

function [predict_function,radialbasis_model] = interpRadialBasisPreModel...
    (X,Y,basis_function)
% radial basis function interp pre model function
% input initial data X,Y,which are real data
% X,Y are x_number x variable_number matrix
% aver_X,stdD_X is 1 x x_number matrix
% output is a radial basis model,include X,Y,base_function
% and predict_function
%
% Copyright 2023 Adel
%
if nargin < 3
    basis_function = [];
end

[x_number,variable_number] = size(X);

% normalize data
aver_X = mean(X);
stdD_X = std(X);
aver_Y = mean(Y);
stdD_Y = std(Y);
index__ = find(stdD_X == 0);
if ~isempty(index__),stdD_X(index__) = 1;end
index__ = find(stdD_Y == 0);
if ~isempty(index__),stdD_Y(index__) = 1;end
X_nomlz = (X-aver_X)./stdD_X;
Y_nomlz = (Y-aver_Y)./stdD_Y;

if isempty(basis_function)
    c = (prod(max(X_nomlz)-min(X_nomlz))/x_number)^(1/variable_number);
    basis_function = @(r) exp(-(r.^2)/c);
end

% initialization distance of all X
X_dis = zeros(x_number,x_number);
for variable_index = 1:variable_number
    X_dis = X_dis+(X_nomlz(:,variable_index)-X_nomlz(:,variable_index)').^2;
end
X_dis = sqrt(X_dis);

[beta,rdibas_matrix] = interpRadialBasis...
    (X_dis,Y_nomlz,basis_function,x_number);

% initialization predict function
predict_function = @(X_predict) interpRadialBasisPredictor...
    (X_predict,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
    x_number,variable_number,beta,basis_function);

radialbasis_model.X = X;
radialbasis_model.Y = Y;
radialbasis_model.radialbasis_matrix = rdibas_matrix;
radialbasis_model.beta = beta;

radialbasis_model.aver_X = aver_X;
radialbasis_model.stdD_X = stdD_X;
radialbasis_model.aver_Y = aver_Y;
radialbasis_model.stdD_Y = stdD_Y;
radialbasis_model.basis_function = basis_function;

radialbasis_model.predict_function = predict_function;

% abbreviation:
% num: number,pred: predict,vari: variable
    function [beta,rdibas_matrix] = interpRadialBasis...
            (X_dis,Y,basis_function,x_number)
        % interp polynomial responed surface core function
        % calculation beta
        %
        % Copyright 2022 Adel
        %
        rdibas_matrix = basis_function(X_dis);

        % stabilize matrix
        rdibas_matrix = rdibas_matrix+eye(x_number)*1e-6;

        % solve beta
        beta = rdibas_matrix\Y;
    end

    function [Y_pred] = interpRadialBasisPredictor...
            (X_pred,X_nomlz,aver_X,stdD_X,aver_Y,stdD_Y,...
            x_num,vari_num,beta,basis_function)
        % radial basis function interpolation predict function
        %
        [x_pred_num,~] = size(X_pred);

        % normalize data
        X_pred_nomlz = (X_pred-aver_X)./stdD_X;

        % calculate distance
        X_dis_pred = zeros(x_pred_num,x_num);
        for vari_index = 1:vari_num
            X_dis_pred = X_dis_pred+...
                (X_pred_nomlz(:,vari_index)-X_nomlz(:,vari_index)').^2;
        end
        X_dis_pred = sqrt(X_dis_pred);

        % predict variance
        Y_pred = basis_function(X_dis_pred)*beta;

        % normalize data
        Y_pred = Y_pred*stdD_Y+aver_Y;
    end

end

%% data library
function [fval_list,con_list,coneq_list] = dataLibraryUpdata...
    (data_library_name,model_function,x_list)
% updata data library
% updata format:
% variable_number,fval_number,con_number,coneq_number
% x,fval,con,coneq
%
fval_list = [];
con_list = [];
coneq_list = [];
[x_number,variable_number] = size(x_list);

if ~strcmp(data_library_name(end-3:end),'.txt')
    data_library_name = [data_library_name,'.txt'];
end

% store format
x_format_base = '%.8e ';
fval_format_base = '%.8e ';
x_format = repmat(x_format_base,1,variable_number);
file_optimalSurrogate_output = fopen(data_library_name,'a');
file_result = fopen('result_total.txt','a');

% updata format:
% variable_number,fval_number,con_number,coneq_number
% x,fval,con,coneq
for x_index = 1:x_number
    x = x_list(x_index,:);
    [fval,con,coneq] = model_function(x);
    fval = fval(:);
    con = con(:);
    coneq = coneq(:);
    fval_list = [fval_list;fval(:)'];
    con_list = [con_list;con(:)'];
    coneq_list = [coneq_list;coneq(:)'];

    % write data to txt_optimalSurrogateSADEKTS
    fprintf(file_optimalSurrogate_output,'%d ',variable_number);
    fprintf(file_optimalSurrogate_output,'%d ',length(fval));
    fprintf(file_optimalSurrogate_output,'%d ',length(con));
    fprintf(file_optimalSurrogate_output,'%d ',length(coneq));

    fprintf(file_optimalSurrogate_output,x_format,x);
    fval_format = repmat(fval_format_base,1,length(fval));
    fprintf(file_optimalSurrogate_output,fval_format,fval);
    fval_format = repmat(fval_format_base,1,length(con));
    fprintf(file_optimalSurrogate_output,fval_format,con);
    fval_format = repmat(fval_format_base,1,length(coneq));
    fprintf(file_optimalSurrogate_output,fval_format,coneq);
    fprintf(file_optimalSurrogate_output,'\n');

    % write data to txt_result
    fprintf(file_result,'%d ',variable_number);
    fprintf(file_result,'%d ',length(fval));
    fprintf(file_result,'%d ',length(con));
    fprintf(file_result,'%d ',length(coneq));

    fprintf(file_result,x_format,x);
    fval_format = repmat(fval_format_base,1,length(fval));
    fprintf(file_result,fval_format,fval);
    fval_format = repmat(fval_format_base,1,length(con));
    fprintf(file_result,fval_format,con);
    fval_format = repmat(fval_format_base,1,length(coneq));
    fprintf(file_result,fval_format,coneq);
    fprintf(file_result,'\n');
end

fclose(file_optimalSurrogate_output);
clear('file_optimalSurrogate_output');
fclose(file_result);
clear('file_result');
end

function [x_list,fval_list,con_list,coneq_list] = dataLibraryLoad...
    (data_library_name,low_bou,up_bou)
% load data from data library
% low_bou,up_bou is range of data
% updata format:
% variable_number,fval_number,con_number,coneq_number
% x,fval,con,coneq
%
if nargin < 3
    up_bou = inf;
    if nargin < 2
        low_bou = -inf;
        if nargin < 1
            error('dataLibraryLoad: lack data_library_name');
        end
    end
end

if ~strcmp(data_library_name(end-3:end),'.txt')
    data_library_name = [data_library_name,'.txt'];
end

% updata format:
% variable_number,fval_number,con_number,coneq_number
% x,fval,con,coneq
if exist(data_library_name,'file')==2
    data_list = importdata(data_library_name);
    if ~isempty(data_list)
        % search whether exist point
        x_list = [];
        fval_list = [];
        con_list = [];
        coneq_list = [];

        for data_index = 1:size(data_list,1)
            data = data_list(data_index,:);

            variable_number = data(1);
            fval_number = data(2);
            con_number = data(3);
            coneq_number = data(4);

            base = 5;
            x = data(base:base+variable_number-1);
            judge = sum(x < low_bou)+sum(x > up_bou);
            if ~judge
                x_list = [x_list;x];
                base = base+variable_number;
                fval_list = [fval_list;data(base:base+fval_number-1)];
                base = base+fval_number;
                con = data(base:base+con_number-1);
                if ~isempty(con)
                    con_list = [con_list;con];
                end
                base = base+con_number;
                coneq = data(base:base+coneq_number-1);
                if ~isempty(coneq)
                    coneq_list = [coneq_list;coneq];
                end
            end
        end
    else
        x_list = [];
        fval_list = [];
        con_list = [];
        coneq_list = [];
    end
else
    x_list = [];
    fval_list = [];
    con_list = [];
    coneq_list = [];
end
end

%% LHD
function [X,X_new,distance_min_nomlz] = getLatinHypercube...
    (sample_number,variable_number,X_exist,...
    low_bou,up_bou,cheapcon_function)
% generate sample sequence latin hypercube
% more uniform point distribution by simulating particle motion
% sample number is total point in area
% default low_bou is 0,up_bou is 1,cheapcon_function is []
% low_bou and up_bou is colume vector
% x in x_exist_list,x_list,supply_x_list is row vector
% x_exist_list should meet bou
%
% Copyright 2022 Adel
%
if nargin < 6
    cheapcon_function = [];
    if nargin < 5
        if nargin < 3
            X_exist = [];
            if nargin < 2
                error('getLatinHypercube: lack variable_number');
            end
        end
        low_bou = zeros(1,variable_number);
        up_bou = ones(1,variable_number);
    end
end
iteration_max = 100;

% check x_exist_list if meet boundary
if ~isempty(X_exist)
    index = find(X_exist < low_bou);
    index = [index,find(X_exist > up_bou)];
    if ~isempty(index)
        error('getLatinHypercube: x_exist_list range error');
    end
    if size(X_exist,2) ~= variable_number
        error('getLatinHypercube: x_exist_list variable_number error');
    end
    X_exist_nomlz = (X_exist-low_bou)./(up_bou-low_bou);
else
    X_exist_nomlz = [];
end

% check input
if sample_number < 0
    X = [];
    X_new = [];
    distance_min_nomlz = [];
    return;
end

% check x_new_number
x_new_number = sample_number-size(X_exist,1);
if x_new_number < 0
    X = X_exist;
    X_new = [];
    distance_min_nomlz = getMinDistance(X_exist_nomlz);
    return;
end

low_bou_nomlz = zeros(1,variable_number);
up_bou_nomlz = ones(1,variable_number);

% get initial X_new_nomalize by lhsdesign
X_new_nomlz = rand(x_new_number,variable_number);
distance_min_nomlz = getMinDistance([X_new_nomlz;X_exist_nomlz]);

% x is nomalize,so constraint function should change
if ~isempty(cheapcon_function)
    cheapcon_function = @(x) ...
        max(max(sample_number*cheapcon_function(x.*(up_bou-low_bou)+low_bou)+1,0),[],1);
end

iteration = 0;
fval_list = zeros(x_new_number,1);
gradient_list = zeros(x_new_number,variable_number);
while iteration < iteration_max
    % change each x place by newton methods
    for x_index = 1:x_new_number
        % get gradient
        [fval_list(x_index,1),gradient_list(x_index,:)] = objectFunctionXPlace...
            (X_new_nomlz(x_index,:),[X_new_nomlz(1:x_index-1,:);X_new_nomlz(x_index+1:end,:);X_exist_nomlz],...
            sample_number,variable_number,low_bou_nomlz-0.1/variable_number,up_bou_nomlz+0.1/variable_number,cheapcon_function);
    end

    % normalize fval
    fval_list = fval_list/max(fval_list);
    for x_index = 1:x_new_number
        C = fval_list(x_index,1)*distance_min_nomlz*(1-iteration/iteration_max);
        x = X_new_nomlz(x_index,:)+...
            -gradient_list(x_index,:)/...
            norm(gradient_list(x_index,:))*C;
        x = min(x,up_bou_nomlz);
        x = max(x,low_bou_nomlz);
        X_new_nomlz(x_index,:) = x;
    end

    iteration = iteration+1;
end
distance_min_nomlz = getMinDistance([X_new_nomlz;X_exist_nomlz]);
X_new = X_new_nomlz.*(up_bou-low_bou)+low_bou;
X = [X_new;X_exist];

    function [fval,gradient] = objectFunctionXPlace...
            (x,X_surplus,sample_number,variable_number,low_bou,up_bou,cheapcon_function)
        % function describe distance between X and X_supply
        % x is colume vector and X_surplus is matrix which is num-1 x var
        % low_bou_limit__ and up_bou_limit__ is colume vector
        % variable in colume
        %
        a__ = 10/variable_number;
        a_bou__ = 30/sample_number;

        sign__ = ((x > X_surplus)-0.5)*2;

        xi__ = -a__*(x-X_surplus).*sign__;
        sum_xi__ = sum(xi__,2);
        psi__ = a__*(low_bou-x)*a_bou__;
        zeta__ = a__*(x-up_bou)*a_bou__;

        %         exp_xi__ = exp(xi__);
        exp_sum_xi__ = exp(sum_xi__);
        exp_psi__ = exp(psi__);
        exp_zeta__ = exp(zeta__);

        % get fval
        fval = sum(exp_sum_xi__,1)+...
            sum(exp_psi__+exp_zeta__,2);

        % get gradient
        gradient = sum(-a__*sign__.*exp_sum_xi__,1)+...
            -a__*exp_psi__*a_bou__+...
            a__*exp_zeta__*a_bou__;

        if ~isempty(cheapcon_function)
            fval_con = cheapcon_function(x);
            fval = fval+fval_con;
            [gradient_con] = differ...
                (cheapcon_function,x,fval_con,variable_number);
            gradient = gradient+gradient_con;
        end

        function [gradient] = differ(differ_function,x,fval,variable_number,step)
            % differ function to get gradient and hessian
            % gradient is rank vector
            %
            if nargin < 5
                step = 1e-6;
            end
            fval__ = zeros(variable_number,2); % backward is 1,forward is 2
            gradient = zeros(1,variable_number);

            % fval and gradient
            for variable_index__ = 1:variable_number
                x_forward__ = x;
                x_backward__ = x;
                x_backward__(variable_index__) = x_backward__(variable_index__)-step;
                fval__(variable_index__,1) = differ_function(x_backward__);

                x_forward__(variable_index__) = x_forward__(variable_index__)+step;
                fval__(variable_index__,2) = differ_function(x_forward__);

                gradient(variable_index__) = ...
                    (fval__(variable_index__,2)-fval__(variable_index__,1))/2/step;
            end
        end
    end
    function distance_min__ = getMinDistance(x_list__)
        % get distance min from x_list
        %

        % sort x_supply_list_initial to decrese distance calculate times
        x_list__ = sortrows(x_list__,1);
        sample_number__ = size(x_list__,1);
        variable_number__ = size(x_list__,2);
        distance_min__ = variable_number__;
        for x_index__ = 1:sample_number__
            x_curr__ = x_list__(x_index__,:);
            x_next_index__ = x_index__ + 1;
            % first dimension only search in min_distance
            search_range__ = variable_number__;
            while x_next_index__ <= sample_number__ &&...
                    (x_list__(x_next_index__,1)-x_list__(x_index__,1))^2 ...
                    < search_range__
                x_next__ = x_list__(x_next_index__,:);
                distance_temp__ = sum((x_next__-x_curr__).^2);
                if distance_temp__ < distance_min__
                    distance_min__ = distance_temp__;
                end
                if distance_temp__ < search_range__
                    search_range__ = distance_temp__;
                end
                x_next_index__ = x_next_index__+1;
            end
        end
        distance_min__ = sqrt(distance_min__);
    end
end
