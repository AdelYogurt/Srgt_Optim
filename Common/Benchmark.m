classdef Benchmark
    % abbreviation:
    % obj: obj, nonlcon: nonlinear constraint, func: function, ...
    % par: parameter, vari: variable, num: number, bou: boundary
    %
    methods
        function self=Benchmark()
            % initialization Test_Function where some functions were sort in
%             global initial_flag
%             fullpath=mfilename('fullpath');
%             [path,name]=fileparts(fullpath);
%             addpath([path,'\Test_Function']);
%             initial_flag=0;
        end

        function [objcon_fcn,vari_num,low_bou,up_bou,obj_fcn,Aineq,Bineq,Aeq,Beq,nonlcon_fcn,x_best,obj_best]=get(self,benchmark_type,benchmark_name)
            % obtain benchmark function by benchmark type and name
            %
            obj_fcn=str2func(['@(x) Benchmark.',benchmark_type,benchmark_name,'Obj(x)']);
            if find(strcmp([benchmark_type,benchmark_name,'Nonlcon'],methods(self)))
                nonlcon_fcn=str2func(['@(x) Benchmark.',benchmark_type,benchmark_name,'Nonlcon(x)']);
            else
                nonlcon_fcn=[];
            end

            % obtain Par
            parameter_function=['self.',benchmark_type,benchmark_name,'Par'];
            [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=eval(parameter_function);

            % packup into objcon_fcn
            if isempty(nonlcon_fcn) && isempty(Aineq) && isempty(Bineq) && isempty(Aeq) && isempty(Beq)
                objcon_fcn=@(x) Benchmark.objconFcn(x,obj_fcn,[]);
            else
                objcon_fcn=@(x) Benchmark.objconFcn(x,obj_fcn,@(x) Benchmark.conFcn(x,Aineq,Bineq,Aeq,Beq,nonlcon_fcn));
            end
        end

        function [MF_objcon_fcn,vari_num,low_bou,up_bou,...
                obj_fcn,Aineq,Bineq,Aeq,Beq,nonlcon_fcn,...
                x_best,obj_best]=getMF(self,benchmark_type,benchmark_name,benchmark_error)
            % obtain benchmark function by benchmark type and name
            %
            obj_fcn=str2func(['@(x) Benchmark.',benchmark_type,benchmark_name,'Obj(x)']);
            if find(strcmp([benchmark_type,benchmark_name,'Nonlcon'],methods(self)))
                nonlcon_fcn=str2func(['@(x) Benchmark.',benchmark_type,benchmark_name,'Nonlcon(x)']);
            else
                nonlcon_fcn=[];
            end

            % obtain Par
            parameter_function=['self.',benchmark_type,benchmark_name,'Par'];
            [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=eval(parameter_function);

            % packup into MF_model
            if isempty(nonlcon_fcn) && isempty(A) && isempty(B) && isempty(Aeq) && isempty(Beq)
                HF_objcon_fcn=@(x) Benchmark.objconFcn(x,obj_fcn,[]);
            else
                HF_objcon_fcn=@(x) Benchmark.objconFcn(x,obj_fcn,@(x) Benchmark.conFcn(x,Aineq,Bineq,Aeq,Beq,nonlcon_fcn));
            end
            
            LF_objcon_fcn=@(x) Benchmark.modelFcnLF(HF_objcon_fcn,x,benchmark_error);
            MF_objcon_fcn={HF_objcon_fcn,LF_objcon_fcn};
        end
    end

    methods(Static) % unconstraint single Objective function
        % vari_num=1
        function obj=singleForresterObj(x)
            % Forrester function
            %
            obj=((x.*6-2).^2).*sin(x.*12-4);
        end
        function obj=singleForresterObjLF(x)
            A=0.5;B=10;C=-5;
            obj=A*((x.*6-2).^2).*sin(x.*12-4)+B*(x-0.5)+C;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleForresterPar()
            vari_num=1;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=0;
            up_bou=1;
            x_best=0.7572;
            obj_best=-6.0207;
        end

        % vari_num=2
        function obj=singleSEObj(x)
            % Sasena function
            %
            x1=x(:,1);x2=x(:,2);
            obj=2+0.01*(x2-x1.^2).^2+(1-x1).^2+2*(2-x2).^2+7*sin(0.5*x1).*sin(0.7*x1.*x2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleSEPar()
            vari_num=2;low_bou=[0,0];up_bou=[5,5];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[2.5044,2.5778];obj_best=-1.4565;
        end

        function obj=singleHIMObj(x)
            % HIM problem
            %
            x1=x(:,1);
            x2=x(:,2);
            obj=(x1.^2+x2-11).^2+(x2.^2+x1+20).^2;
        end
        function obj=singleHIMObjLF(x)
            x1=x(:,1);
            x2=x(:,2);
            obj=(0.9*x1.^2+0.8*x2-11).^2+(0.8*x2.^2+0.9*x1+20).^2-(x1+1).^2;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleHIMPar()
            vari_num=2;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[-5,-5];
            up_bou=[5,5];
            x_best=[-3.6483,-0.0685];
            obj_best=272.5563;
        end

        function obj=singleRSObj(x)
            % Rastrigin function
            % multi local minimum function
            %
            x1=x(:,1);x2=x(:,2);
            obj=x1.^2+x2.^2-cos(18*x1)-cos(18*x2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleRSPar()
            vari_num=2;low_bou=[-1,-1];up_bou=[1,1];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[0,0];obj_best=-2;
        end
        
        function obj=singleROS2DObj(x)
            % 2D Rosenbrock function
            %
            obj=sum(100*(x(2:2)-x(1:1).^2).^2+(x(1:1)-1).^2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleROS2DPar()
            vari_num=2;low_bou=zeros(1,2);up_bou=ones(1,2);
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[1,1];obj_best=0;
        end


        function obj=singleBRObj(x)
            % Branin function
            %
            x1=x(1);
            x2=x(2);
            obj=(x2-5.1/4/pi/pi*x1^2+5/pi*x1-6)^2+10*(1-1/8/pi)*cos(x1)+10;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleBRPar()
            vari_num=2;low_bou=[-5,10];up_bou=[0,15];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[-3.1416,12.2750];obj_best=0.3979;
        end

        function obj=singleGFObj(x)
            % Generalized polynomial function 
            %
            c1=1.5;c2=2.25;c3=2.625;
            x1=x(:,1);x2=x(:,2);
            u1=c1-x1.*(1-x2);
            u2=c2-x1.*(1-x2.^2);
            u3=c3-x1.*(1-x2.^3);
            obj=u1.^2+u2.^2+u3.^2;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleGFPar()
            vari_num=2;low_bou=[-5,-5];up_bou=[5,5];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[3.0000,0.5000];obj_best=0;
        end

        function obj=singleGPObj(x)
            x1=x(:,1);x2=x(:,2);
            obj=(1+(x1+x2+1).^2.*...
                (19-14*x1+3*x1.^2-14*x2+6*x1.*x2+3*x2.^2)).*...
                (30+(2*x1-3*x2).^2.*(18-32*x1+12*x1.^2+48*x2-36*x1.*x2+27*x2.^2));
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleGPPar()
            vari_num=2;low_bou=[-2,-2];up_bou=[2,2];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[0,-1];obj_best=3;
        end

        function obj=singlePKObj(x)
            x1=x(:,1);
            x2=x(:,2);
            obj=3*(1-x1).^2.*exp(-(x1.^2)-(x2+1).^2) ...
                -10*(x1/5-x1.^3-x2.^5).*exp(-x1.^2-x2.^2) ...
                -1/3*exp(-(x1+1).^2-x2.^2) ;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singlePKPar()
            vari_num=2;low_bou=[-3,-3];up_bou=[3,3];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[0.2283,-1.6255]; obj_best=-6.5511;
        end

        function obj=singleSCObj(x)
            x1=x(:,1);
            x2=x(:,2);
            obj=4*x1.^2-2.1*x1.^4+x1.^6/3+x1.*x2-4*x2.^2+4*x2.^4;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleSCPar()
            vari_num=2;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[-2,-2];
            up_bou=[2,2];
            x_best=[0.0898,-0.7127]; 
            obj_best=-1.0316;
        end

        % vari_num=4
        function obj=singleCOLObj(x)
            x1=x(:,1);
            x2=x(:,2);
            x3=x(:,3);
            x4=x(:,4);
            obj=100*(x1.^2-x2).^2+(x1-1).^2+(x3-1).^2+90*(x3.^2-x4).^2+...
                10.1*((x2-1).^2-(x4-1).^2)+19.8*(x2-1).*(x4-1);
        end
        function obj=singleCOLObjLF(x)
            obj=func_H2_COL_H([0.8,0.8,0.5,0.5].*x);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleCOLPar()
            vari_num=4;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=zeros(1,vari_num);
            up_bou=ones(1,vari_num);
            x_best=[1.0000,1.0000,0.1667,0.0000];
            obj_best=-9.3361;
        end

        function obj=singleROS4DObj(x)
            % modified Rosenbrock problem
            %
            obj=sum((100*(x(2:4)-x(1:3).^2).^2-(x(1:3)-1).^2).^2);
        end
        function obj=singleROS4DObjLF(x)
            obj=sum((100*(0.5*x(2:4)-0.6*x(1:3).^2).^2-(0.5*x(1:3)-0.5).^2).^2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleROS4DPar()
            vari_num=4;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[-2,-2,-2,-2];
            up_bou=[2,2,2,2];
            x_best=[0.2131,-0.0333,-0.1022,0.1207];
            obj_best=0;
        end

        % vari_num=5
        function obj=singleST5Obj(x)
            s=[0.28,0.59,0.47,0.16,0.32];
            x=x-s;
            obj=0.5*sum(x.^4-16*x.^2+5*x);
        end
        function obj=singleST5ObjLF(x)
            obj=0.5*sum(x.^4-16*x.^2+5*x);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleST5Par()
            vari_num=5;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[-5,-5,-5,-5,-5];
            up_bou=[5,5,5,5,5];
            x_best=[-2.6235,-2.3135,-2.4335,-2.7435,-2.5835];
            obj_best=-195.8308;
        end

        % vari_num=6
        function obj=singleHNObj(x)
            % Hartman function
            %
            coe=[
                1 10   3   17   3.5 1.7 8  1   0.1312 0.1696 0.5569 0.0124 0.8283 0.5886;
                2 0.05 10  17   0.1 8   14 1.2 0.2329 0.4135 0.8307 0.3736 0.1004 0.9991;
                3 3    3.5 1.7  10  17  8  3   0.2348 0.1451 0.3522 0.2883 0.3047 0.6650;
                4 17   8   0.05 10  0.1 14 3.2 0.4047 0.8828 0.8732 0.5743 0.1091 0.0381;];

            alpha=coe(:,2:7)';
            c=coe(:,8);
            p=coe(:,9:14);

            obj=0;
            for i=1:4
                hari=(x-p(i,:)).^2*alpha(:,i);
                obj=c(i)*exp(-hari)+obj;
            end
            obj=-obj;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleHNPar()
            vari_num=6;low_bou=zeros(1,6);up_bou=ones(1,6);
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[0.2017;0.1500;0.4769;0.2753;0.3117;0.6573];obj_best=-3.3224;
        end

        % vari_num=10
        function obj=singleROS10DObj(x)
            % 10D Rosenbrock function
            %
            obj=sum(100*(x(2:10)-x(1:9).^2).^2+(x(1:9)-1).^2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleROS10DPar()
            vari_num=10;low_bou=zeros(1,10);up_bou=ones(1,10);
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[1,1,1,1,1,1,1,1,1,1];obj_best=0;
        end

        function obj=singleA10Obj(x)
            s=[1.3;0.1;1.4;0.8;1.7;1;1.5;0.6;2;0.4]';
            obj=-20*exp(-0.2*sqrt(sum((x-s).^2,2)/10))-exp(sum(cos(2*1.3*pi*(x-s))/10,2))+20+exp(1);
        end
        function obj=singleA10ObjLF(x)
            obj=-20*exp(-0.2*sqrt(sum(x.^2,2)/10))-exp(sum(cos(2*pi*x)/10,2))+20+exp(1);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleA10Par()
            vari_num=10;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[0,0,0,0,0,0,0,0,0,0];
            up_bou=[1,1,1,1,1,1,1,1,1,1];
            x_best=[0.5608,0.1000,0.6608,0.8000,0.9608,0.9997,0.7608,0.6000,1.0000,0.4000];
            obj_best=2.4968;
        end

        function obj=singleF16Obj(x)
            AM=[1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,1;
                0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,0;
                0,0,1,0,0,0,1,0,1,1,0,0,0,1,0,0;
                0,0,0,1,0,0,1,0,0,0,1,0,0,0,1,0;
                0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1;
                0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0;
                0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0;
                0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0;
                0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1;
                0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0;
                0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0;
                0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0;
                0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0;
                0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0;
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0;
                0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1;];
            x_Max=x.^2+x+1;obj=zeros(size(x,1),1);
            for i=1:16
                for j=1:16
                    obj=obj+AM(i,j).*x_Max(:,i).*x_Max(:,j);
                end
            end
        end 
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleF16Par()
            vari_num=16;low_bou=-1*ones(1,vari_num);up_bou=zeros(1,vari_num);
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[0.5608,0.1000,0.6608,0.8000,0.9608,0.9997,0.7608,0.6000,1.0000,0.4000];
            obj_best=2.4968;
        end

        % vari_num=20
        function obj=singleA20Obj(x)
            s=[1.3;0.1;1.4;0.8;1.7;1;1.5;0.6;2;0.4;1.3;0.3;1.5;0.9;1.9;1.1;1.7;0.7;2.1;0.5]';
            obj=-20*exp(-0.2*sqrt(sum((x-s).^2,2)/10))-exp(sum(cos(2*1.3*pi*(x-s))/10,2))+20+exp(1);
        end
        function obj=singleA20ObjLF(x)
            obj=-20*exp(-0.2*sqrt(sum(x.^2,2)/10))-exp(sum(cos(2*pi*x)/10,2))+20+exp(1);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleA20Par()
            vari_num=20;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
            up_bou=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1];
            x_best=[0.5357,0.1000,0.6357,0.8000,0.9357,1.0000,0.7357,0.6000,0.4714,0.4000,0.5357,0.3000,0.7357,0.9000,0.3714,1.0000,0.9357,0.7000,0.5714,0.5000];
            obj_best=-0.6291;
        end

        function obj=singleDP20Obj(x)
            s=[1.8,0.5,2,1.2,0.4,0.2,1.4,0.3,1.6,0.6,0.8,1,1.3,1.9,0.7,1.6,0.3,1.1,2,1.4];
            x=x-s;
            obj=(x(:,1)-1).^2+sum((2:20).*(2*x(:,2:end).^2-x(:,1:end-1)).^2,2);
        end
        function obj=singleDP20ObjLF(x)
            obj=(x(:,1)-1).^2+sum((2:20).*(2*x(:,2:end).^2-x(:,1:end-1)).^2,2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleDP20Par()
            vari_num=20;low_bou=ones(1,vari_num)*-30;up_bou=ones(1,vari_num)*30;
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[2.1333,0.5000,2.0000,1.2000,0.4000,0.2000,1.4000,0.3000,1.6000,0.6000,0.8000,1.0000,1.3000,1.9000,0.7000,1.6000,0.3009,1.1214,2.1035,1.1725];
            obj_best=0.6667;
        end
        
        function obj=singleEP20Obj(x)
            ss=[1.8,0.4,2,1.2,1.4,0.6,1.6,0.2,0.8,1,1.3,1.1,2,1.4,0.5,0.3,1.6,0.7,0.3,1.9];
            sh=[0.3,0.4,0.2,0.6,1,0.9,0.2,0.8,0.5,0.7,0.4,0.3,0.7,1,0.9,0.6,0.2,0.8,0.2,0.5];
            obj=sum((1:20).*sh.*(x-ss).^2,2);
        end
        function obj=singleEP20ObjLF(x)
            obj=sum((1:20).*x.^2,2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleEP20Par()
            vari_num=20;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=ones(1,vari_num)*-30;
            up_bou=ones(1,vari_num)*30;
            x_best=[1.8000,0.4000,2.0000,1.2000,1.4000,0.6000,1.6000,0.2000,0.8000,1.0000,1.3000,1.1000,2.0000,1.4000,0.5000,0.3000,1.6000,0.7000,0.3000,1.9000];
            obj_best=0;
        end
 
        % vari_num=30
        function obj=singleAckley30Obj(x)
            % Ackley problem 30 dimension
            %
            n=30;
            sum1=0;
            sum2=0;

            for i=1:n
                sum1=sum1+x(i)^2;
                sum2=sum2+cos((2*pi)*x(i));
            end

            obj=-20*exp(-0.2*sqrt(1/n*sum1))-exp(1/n*sum2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleAckley30Par()
            vari_num=30;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=-15*ones(1,vari_num);
            up_bou=20*ones(1,vari_num);
            x_best=zeros(1,vari_num);
            obj_best=-20-exp(1);
        end

    end
    
    methods(Static) % constraint single Objective function
        % vari_num=2
        function [con,coneq]=singleHauptNonlcon(x)
            x1=x(:,1);
            x2=x(:,2);
            coneq=[];
            con=[x1.*sin(4*x1)+1.1*x2*sin(2*x2),-x1-x2+3];
            con(:,1)=con(:,1)+1.5;
        end
        function obj=singleHauptObj(x)
            % Haupt function
            %
            x1=x(:,1);
            x2=x(:,2);
            obj=(x1-3.7).^2+(x2-4).^2;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleHauptPar()
            vari_num=2;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[0,0];
            up_bou=[3.7,4];
            x_best=[2.9711,3.4035];
            obj_best=0.8871;
        end

        function [con,coneq]=singleG06Nonlcon(x)
            g1=-(x(:,1)-5).^2-(x(:,2)-5).^2+100;
            g2=(x(:,1)-6).^2+(x(:,2)-5).^2-82.81;
            con=[g1,g2];
            coneq=[];
        end
        function obj=singleG06Obj(x)
            obj=(x(:,1)-10).^3+(x(:,2)-20).^3;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG06Par()
            vari_num=2;low_bou=[13,0];up_bou=[100,100];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[14.0950,0.8430];obj_best=-6.9618e+03;
        end

        % vari_num=4
        function [con,coneq]=singlePVDNonlcon(x)
            x1=x(1);x2=x(2);x3=x(3);x4=x(4);
            con=-pi*x3^2.*x4-4/3*pi*x3^3+1296000;
            coneq=[];
        end
        function obj=singlePVDObj(x)
            x1=x(1);x2=x(2);x3=x(3);x4=x(4);
            obj=0.6224*x1*x3*x4+1.7781*x2*x3^2+3.1661*x1^2*x4+19.84*x1^2*x3;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singlePVDPar()
            vari_num=4;low_bou=[1,0.625,25,25];up_bou=[1.375,1,150,240];
            Aineq=[ -1,  0,  0.0193, 0;
                 0, -1, 0.00954, 0;
                 0,  0,       0, 1;
                -1,  0,       0, 0;
                 0, -1,       0, 0];
            Bineq=[0;0;240;-1.1;-0.6];
            Aeq=[];Beq=[];
            x_best=[1.1000,0.6250,56.9948,51.0013];
            obj_best=7.1637e+03;
        end

        function [con,coneq]=singlePVD4Nonlcon(x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);
            g3=-pi*x3.^2.*x4-4/3*pi*x3.^3+1296000;
            boolean=g3 >= 0;
            g3(boolean)=log(1+g3(boolean));
            g3(~boolean)=-log(1-g3(~boolean));
            con=g3;
            coneq=[];
        end
        function obj=singlePVD4Obj(x)
            % Pressure vessel design (PVD4) problem
            %
            x1=x(1);x2=x(2);x3=x(3);x4=x(4);
            obj=0.6224*x1*x3*x4+1.7781*x2*x3^2+3.1661*x1^2*x4+19.84*x1^2*x3;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singlePVD4Par()
            vari_num=4;
            Aineq=[-1,0,0.0193,0;
                0,-1,0.00954,0;];
            Bineq=[0;0];
            Aeq=[];
            Beq=[];
            low_bou=[0,0,0,0];
            up_bou=[1,1,50,240];
            x_best=[0.7276,0.3596,37.6991,240.0000];
            obj_best=5804.45;
        end

        function [con,coneq]=singleG05MODNonlcon(x)
            coneq=[];
            con(:,1)=x(:,3)-x(:,4)-0.55;
            con(:,2)=x(:,4)-x(:,3)-0.55;
            con(:,3)=1000*sin(-x(:,3)-0.25)+1000*sin(-x(:,4)-0.25)+894.8-x(:,1);
            con(:,4)=1000*sin(x(:,3)-0.25)+1000*sin(x(:,3)-x(:,4)-0.25)+894.8-x(:,2);
            con(:,5)=1000*sin(x(:,4)-0.25)+1000*sin(x(:,4)-x(:,3)-0.25)+1294.8;
        end
        function obj=singleG05MODObj(x)
            obj=3*x(:,1)+1e-6*x(:,1).^3+2*x(:,2)+2e-6/3*x(:,2).^3;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG05MODPar()
            vari_num=4;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[0, 0, -0.55, -0.55];
            up_bou=[1200, 1200, 0.55, 0.55];
            x_best=[0.679945319505338   1.026067132980041   0.000118876364740  -0.000396233553086]*1e3;
            obj_best=5.1265e+03;
        end

        % vari_num=5
        function [con,coneq]=singleICENonlcon(x)
            b=x(1);
            c_r=x(2);
            d_E=x(3);
            d_I=x(4);
            omega=x(5)*1e-3;

            K_1=1.2;
            K_2=2;
            K_3=0.82;
            K_4=(-1e-12+30.99)/37.34; K_5=0.89;
            K_6=0.6;
            K_7=6.5;
            K_8=230.5;
            L_1=400;
            L_2=200;
            rou=1.225;
            gama=1.33;
            V=1.859*1e6;
            Q=43958;
            N_c=4;
            C_s=0.44;
            A_f=14.6;

            S_V=0.83*((8+4*c_r)+1.5*(c_r-1)*(pi*N_c/V)*b^3)/((2+c_r)*b);
            eta_tw=0.8595*(1-c_r^(-0.33))-S_V;

            g1=K_1*N_c*b-L_1;
            g2=(4*K_2*V/pi/N_c/L_2)^0.5-b;
            g3=d_I+d_E-K_3*b;
            g4=K_4*d_I-d_E;
            g5=d_E-K_5*d_I;
            g6=9.428*1e-5*(4*V/pi/N_c)*(omega/d_I^2)-K_6*C_s;
            g7=c_r-13.2+0.045*b;
            g8=omega-K_7;
            g9=3.6*1e6-K_8*Q*eta_tw;

            con=[g1,g2,g3,g4,g5,g6,g7,g8,g9];
            coneq=[];
        end
        function obj=singleICEObj(x)
            b=x(1);
            c_r=x(2);
            d_E=x(3);
            d_I=x(4);
            omega=x(5)*1e-3;

            K_0=1/120;
            rou=1.225;
            gama=1.33;
            V=1.859*1e6;
            Q=43958;
            N_c=4;
            C_s=0.44;
            A_f=14.6;

            if omega > 5.25
                eta_vb=1.067-0.038*exp(omega-5.25);
            else
                eta_vb=0.637+0.13*omega-0.014*omega^2+0.00066*omega^3;
            end
            eta_tad=0.8595*(1-c_r^(-0.33));
            eta_V=eta_vb*(1+5.96*1e-3*omega^2)/(1+((9.428*1e-5)*(4*V/pi/N_c/C_s)*(omega/d_I^2))^2);
            S_V=0.83*((8+4*c_r)+1.5*(c_r-1)*(pi*N_c/V)*b^3)/((2+c_r)*b);
            eta_t=eta_tad-S_V*(1.5/omega)^0.5;
            V_P=(8*V/pi/N_c)*omega*b^(-2);
            FMEP=4.826*(c_r-9.2)+(7.97+0.253*V_P+9.7*(1e-6)*V_P^2);

            obj=K_0*(FMEP-(rou*Q/A_f)*eta_t*eta_V)*omega;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleICEPar()
            vari_num=5;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[75,6,25,35,5000];
            up_bou=[90,12,35,45,10000];
            x_best=[83.33,9.45,30.99,37.34,6070];
            obj_best=-55.67;
        end

        function [con,coneq]=singleG04Nonlcon(x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);x5=x(:,5);

            g1=85.334407+0.0056858*x2.*x5+0.0006262*x1.*x4-0.0022053*x3.*x5-92;
            g2=-85.334407-0.0056858*x2.*x5-0.0006262*x1.*x4+0.0022053*x3.*x5;
            g3=80.51249+0.0071317*x2.*x5+0.0029955*x1.*x2+0.0021813*x3.^2-110;
            g4=-80.51249-0.0071317*x2.*x5-0.0029955*x1.*x2-0.0021813*x3.^2+90;
            g5=9.300961+0.0047026*x3.*x5+0.0012547*x1.*x3+0.0019085*x3.*x4-25;
            g6= -9.300961-0.0047026*x3.*x5-0.0012547*x1.*x3-0.0019085*x3.*x4+20;

            con=[g1,g2,g3,g4,g5,g6];
            coneq=[];
        end
        function obj=singleG04Obj(x)
            x1=x(:,1);x2=x(:,2);x3=x(:,3);x4=x(:,4);x5=x(:,5);
            obj=5.3578547*x3.^2+0.8356891*x1.*x5+37.293239*x1-40792.141;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG04Par()
            vari_num=5;low_bou=[78,33,27,27,27];up_bou=[102,45,45,45,45];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[78,33,29.9952560256815985,45,36.7758129057882073];obj_best=-3.066553867178332e4;
        end

        function [con,coneq]=singleG16Nonlcon(x)
            coneq=[];
            y1=x(:, 2)+x(:, 3)+41.6;
            c1=0.024 * x(:, 4)-4.62;
            y2=12.5./c1+12;
            c2=0.0003535 * x(:, 1).^2+0.5311 * x(:, 1)+0.08705 * y2.* x(:, 1);
            c3=0.052 * x(:, 1)+78+0.002377 * y2.* x(:, 1);
            y3=c2./c3;
            y4=19 * y3;
            c4=0.04782 * (x(:, 1)-y3)+0.1956 * (x(:, 1)-y3).^2./x(:, 2)+0.6376 * y4+1.594 * y3;
            c5=100 * x(:, 2);
            c6=x(:, 1)-y3-y4;
            c7=0.950-c4./c5;
            y5=c6.* c7;
            y6=x(:, 1)-y5-y4-y3;
            c8=(y5+y4) * 0.995;
            y7=c8./y1;
            y8=c8/3798;
            c9=y7-0.0663 * y7./y8-0.3153;
            y9=96.82./c9+0.321 * y1;
            y10=1.29 * y5+1.258 * y4+2.29 * y3+1.71 * y6;
            y11=1.71 * x(:, 1)-0.452 * y4+0.580 * y3;
            c10=12.3 / 752.3;
            c11=1.75 * y2.* 0.995.* x(:, 1);
            c12=0.995 * y10+1998.0;
            y12=c10 * x(:, 1)+(c11./ c12);
            y13=c12-1.75 * y2;
            y14=3623.0+64.4 * x(:, 2)+58.4 * x(:, 3)+(146312.0./ (y9+x(:, 5)));
            c13=0.995 * y10+60.8 * x(:, 2)+48 * x(:, 4)-0.1121 * y14-5095.0;
            y15=y13./ c13;
            y16=148000.0-331000.0 * y15+40.0 * y13-61.0 .* y15 .* y13;
            c14=2324 * y10-28740000 * y2;
            y17=14130000-1328.0 * y10-531.0 * y11+(c14./c12);
            c15=(y13./y15)-(y13/ 0.52);
            c16=1.104-0.72 * y15;
            c17=y9+x(:, 5);

            con(:, 1)=0.28/0.72.* y5-y4;
            con(:, 2)=x(:, 3)-1.5 * x(:, 2);
            con(:, 3)=3496.* y2./c12-21;
            con(:, 4)=110.6+y1-62212./c17;
            con(:, 5)=213.1-y1;
            con(:, 6)=y1-405.23;
            con(:, 7)=17.505-y2;
            con(:, 8)=y2-1053.6667;
            con(:, 9)=11.275-y3;
            con(:, 10)=y3-35.03;
            con(:, 11)=214.228-y4;
            con(:, 12)=y4-665.585;
            con(:, 13)=7.458-y5;
            con(:, 14)=y5-584.463;
            con(:, 15)=0.961-y6;
            con(:, 16)=y6-265.916;
            con(:, 17)=1.612-y7;
            con(:, 18)=y7-7.046;
            con(:, 19)=0.146-y8;
            con(:, 20)=y8-0.222;
            con(:, 21)=107.99-y9;
            con(:, 22)=y9-273.366;
            con(:, 23)=922.693-y10;
            con(:, 24)=y10-1286.105;
            con(:, 25)=926.832-y11;
            con(:, 26)=y11-1444.046;
            con(:, 27)=18.766-y12;
            con(:, 28)=y12-537.141;
            con(:, 29)=1072.163-y13;
            con(:, 30)=y13-3247.039;
            con(:, 31)=8961.448-y14;
            con(:, 32)=y14-26844.086;
            con(:, 33)=0.063-y15;
            con(:, 34)=y15-0.386;
            con(:, 35)=71084.33-y16;
            con(:, 36)=-140000+y16;
            con(:, 37)=2802713-y17;
            con(:, 38)=y17-12146108;

        end
        function obj=singleG16Obj(x)
            y1=x(:, 2)+x(:, 3)+41.6;
            c1=0.024 * x(:, 4)-4.62;
            y2=12.5./c1+12;
            c2=0.0003535 * x(:, 1).^2+0.5311 * x(:, 1)+0.08705 * y2.* x(:, 1);
            c3=0.052 * x(:, 1)+78+0.002377 * y2.* x(:, 1);
            y3=c2./c3;
            y4=19 * y3;
            c4=0.04782 * (x(:, 1)-y3)+0.1956 * (x(:, 1)-y3).^2./x(:, 2)+0.6376 * y4+1.594 * y3;
            c5=100 * x(:, 2);
            c6=x(:, 1)-y3-y4;
            c7=0.950-c4./c5;
            y5=c6.* c7;
            y6=x(:, 1)-y5-y4-y3;
            c8=(y5+y4) * 0.995;
            y7=c8./y1;
            y8=c8/3798;
            c9=y7-0.0663 * y7./y8-0.3153;
            y9=96.82./c9+0.321 * y1;
            y10=1.29 * y5+1.258 * y4+2.29 * y3+1.71 * y6;
            y11=1.71 * x(:, 1)-0.452 * y4+0.580 * y3;
            c10=12.3 / 752.3;
            c11=1.75 * y2.* 0.995.* x(:, 1);
            c12=0.995 * y10+1998.0;
            y12=c10 * x(:, 1)+(c11./ c12);
            y13=c12-1.75 * y2;
            y14=3623.0+64.4 * x(:, 2)+58.4 * x(:, 3)+(146312.0./ (y9+x(:, 5)));
            c13=0.995 * y10+60.8 * x(:, 2)+48 * x(:, 4)-0.1121 * y14-5095.0;
            y15=y13./ c13;
            y16=148000.0-331000.0 * y15+40.0 * y13-61.0 .* y15 .* y13;
            c14=2324 * y10-28740000 * y2;
            y17=14130000-1328.0 * y10-531.0 * y11+(c14./c12);
            c15=(y13./y15)-(y13/ 0.52);
            c16=1.104-0.72 * y15;
            c17=y9+x(:, 5);

            obj=0.000117 * y14+0.1365+0.00002358 * y13+0.000001502 * y16+0.0321 * y12 ...
               +0.004324 * y5+0.0001 * (c15 ./ c16)+37.48 * (y2./c12)-0.0000005843 * y17;

        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG16Par()
            vari_num=5;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[704.4148,68.6,0,193,25];
            up_bou=[906.3855,288.88,134.75,287.0966,84.1988];
            x_best=[705.174537070090537,68.5999999999999943,102.899999999999991,282.324931593660324,37.5841164258054832];
            obj_best=-1.90515525853479;
        end

        % vari_num=7
        function [con,coneq]=singleSR7Nonlcon(x)
            coneq=[];
            matTemp1=sqrt((745*x(:,4)./(x(:,2).*x(:,3))).^2+16.91e6);
            matTemp2=sqrt((745*x(:,5)./(x(:,2).*x(:,3))).^2+157.5e6);
            con(:,1)=(27-x(:,1).*x(:,2).^2.*x(:,3) ...
                ) / 27;
            con(:,2)=(397.5-x(:,1).*x(:,2).^2 ...
                .*x(:,3).^2) / 397.5;
            con(:,3)=(1.93-(x(:,2).*x(:,6).^4 ...
                .*x(:,3) ./ (x(:,4).^3))) / 1.93;
            con(:,4)=(1.93-(x(:,2).*x(:,7).^4 ...
                .*x(:,3) ./ (x(:,5).^3))) / 1.93;
            con(:,5)=(matTemp1 ./ (0.1*x(:,6).^3)-1100) / 1100;
            con(:,6)=(matTemp2 ./ (0.1*x(:,7).^3)-850) / 850;
            con(:,7)=(x(:,2).*x(:,3)-40) / 40;
            con(:,8)=(5-x(:,1)./x(:,2)) / 5;
            con(:,9)=(x(:,1)./x(:,2)-12) / 12;
            con(:,10)=(1.9+1.5*x(:,6)-x(:,4)) / 1.9;
            con(:,11)=(1.9+1.1*x(:,7)-x(:,5)) / 1.9;
        end
        function obj=singleSR7Obj(x)
            matTemp1=3.3333*x(:,3).^2+14.9334*x(:,3)-43.0934;
            matTemp2=x(:,6).^2+x(:,7).^2;
            matTemp3=x(:,6).^3+x(:,7).^3;
            matTemp4=x(:,4).*x(:,6).^2+x(:,5)...
                .*x(:,7).^2;
            obj=0.7854*x(:,1).*x(:,2).^2.*matTemp1 ...
               -1.508*x(:,1).*matTemp2+7.477*matTemp3 ...
               +0.7854*matTemp4;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleSR7Par()
            vari_num=7;low_bou=[2.6, 0.7, 17, 7.3, 7.3, 2.9, 5];up_bou=[3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5];
            Aineq=[];Bineq=[];Aeq=[];Beq=[];
            x_best=[3.5000    0.7000   17.0000    7.3000    7.7153    3.3505    5.2867];obj_best=2.9944e+03;
        end

        % vari_num=7
        function [con,coneq]=singleG09Nonlcon(x)
            coneq=[];
            con(:, 1)=-127+2 * x(:, 1).^2+3 * x(:, 2).^4+x(:, 3)+4 * x(:, 4).^2+5 * x(:, 5);
            con(:, 2)=-282+7 * x(:, 1)+3 * x(:, 2)+10 * x(:, 3).^2+x(:, 4)-x(:, 5);
            con(:, 3)=-196+23 * x(:, 1)+x(:, 2).^2+6 * x(:, 6).^2-8 * x(:, 7);
            con(:, 4)=4 * x(:, 1).^2+x(:, 2).^2-3 * x(:, 1) .* x(:, 2)+2 * x(:, 3).^2+5 * x(:, 6)-11 * x(:, 7);
        end
        function obj=singleG09Obj(x)
            obj=(x(:, 1)-10).^2+5 * (x(:, 2)-12).^2+x(:, 3).^4+3 * (x(:, 4)-11).^2+10 * x(:, 5).^6+7 * x(:, 6).^2+x(:, 7).^4-4 * x(:, 6) .* x(:, 7)-10 * x(:, 6)-8 * x(:, 7);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG09Par()
            vari_num=7;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[-10 -10 -10 -10 -10 -10 -10];
            up_bou=[10 10 10 10 10 10 10];
            x_best=[2.330499020860739   1.951372301776673  -0.477543604918950   4.365726520068069  -0.624486603992181   1.038130162630999  1.594226316347420];
            obj_best=6.806300573697980e+02;
        end

        function [con,coneq]=singleG23Nonlcon(x)
            coneq=[];
            x1=x(:,1);
            x2=x(:,2);
            x3=x(:,3);
            x4=x(:,4);
            x5=x(:,5);
            x6=x(:,6);
            x7=x(:,7);
            x8=x(:,8);
            x9=x(:,9);
            con(:, 1)=x9.*x3+0.02.*x6-0.025.*x5;
            con(:, 2)=x9.*x4+0.02.*x7-0.015.*x8;
            con(:, 3)=x1+x2-x3-x4;
            con(:, 4)=0.03.*x1+0.01.*x2-x9.*(x3+x4);
            con(:, 5)=x3+x6-x5;
            con(:, 6)=x4+x7-x8 ;
        end
        function obj=singleG23Obj(x)
            x1=x(:,1);
            x2=x(:,2);
            x3=x(:,3);
            x4=x(:,4);
            x5=x(:,5);
            x6=x(:,6);
            x7=x(:,7);
            x8=x(:,8);
            x9=x(:,9);
            obj=-9.*x5-15.*x8+6.*x1+16.*x2+10.*(x6+x7);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG23Par()
            vari_num=9;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[0 0 0 0 0 0 0 0 0.01];
            up_bou=[300 300 100 200 100 300 100 200 0.03];
            x_best=[0,0,0,68.8087,100.0000,0,0,200.0000,0.0100];
            obj_best=-3900;
        end

        % vari_num=9
        function [con,coneq]=singleG18Nonlcon(x)
            coneq=[];
            con(:, 1)=x(:, 3).^2+x(:, 4).^2-1;
            con(:, 2)=x(:, 9).^2-1;
            con(:, 3)=x(:, 5).^2+x(:, 6).^2-1;
            con(:, 4)=x(:, 1).^2+(x(:, 2)-x(:, 9)).^2-1;
            con(:, 5)=(x(:, 1)-x(:, 5)).^2+(x(:, 2)-x(:, 6)).^2-1;
            con(:, 6)=(x(:, 1)-x(:, 7)).^2+(x(:, 2)-x(:, 8)).^2-1;
            con(:, 7)=(x(:, 3)-x(:, 5)).^2+(x(:, 4)-x(:, 6)).^2-1;
            con(:, 8)=(x(:, 3)-x(:, 7)).^2+(x(:, 4)-x(:, 8)).^2-1;
            con(:, 9)=x(:, 7).^2+(x(:, 8)-x(:, 9)).^2-1;
            con(:, 10)=x(:, 2).* x(:, 3)-x(:, 1).* x(:, 4);
            con(:, 11)=-x(:, 3).* x(:, 9);
            con(:, 12)=x(:, 5).* x(:, 9);
            con(:, 13)=x(:, 6).* x(:, 7)-x(:, 5).* x(:, 8);

        end
        function obj=singleG18Obj(x)
            obj=-0.5 * (x(:, 1).* x(:, 4)-x(:, 2).* x(:, 3)+x(:, 3).* x(:, 9)-x(:, 5).* x(:, 9)+x(:, 5).* x(:, 8)-x(:, 6).* x(:, 7));
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG18Par()
            vari_num=9;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=[-10,-10,-10,-10,-10,-10,-10,-10,0];
            up_bou=[10,10,10,10,10,10,10,10,20];
            x_best=[-0.657776192427943163,-0.153418773482438542,0.323413871675240938,-0.946257611651304398,-0.657776194376798906,-0.753213434632691414,0.323413874123576972,-0.346462947962331735,0.59979466285217542];
            obj_best=-0.866025403784439;
        end

        % vari_num=10
        function [con,coneq]=singleG07Nonlcon(x)
            coneq=[];
            con(:, 1)=-105+4 * x(:, 1)+5 * x(:, 2)-3 * x(:, 7)+9 * x(:, 8);
            con(:, 2)=10 * x(:, 1)-8 * x(:, 2)-17 * x(:, 7)+2 * x(:, 8);
            con(:, 3)=-8 * x(:, 1)+2 * x(:, 2)+5 * x(:, 9)-2 * x(:, 10)-12;
            con(:, 4)=3 * (x(:, 1)-2).^2+4 * (x(:, 2)-3).^2+2 * x(:, 3).^2-7 * x(:, 4)-120;
            con(:, 5)=5 * x(:, 1).^2+8 * x(:, 2)+(x(:, 3)-6).^2-2 * x(:, 4)-40;
            con(:, 6)=x(:, 1).^2+2 * (x(:, 2)-2).^2-2 * x(:, 1) .* x(:, 2)+14 * x(:, 5)-6 * x(:, 6);
            con(:, 7)=0.5 * (x(:, 1)-8).^2+2 * (x(:, 2)-4).^2+3 * x(:, 5).^2-x(:, 6)-30;
            con(:, 8)=-3 * x(:, 1)+6 * x(:, 2)+12 * (x(:, 9)-8).^2-7 * x(:, 10);
        end
        function obj=singleG07Obj(x)
            obj=x(:, 1).^2+x(:, 2).^2+x(:, 1) .* x(:, 2)-14 * x(:, 1)-16 * x(:, 2)+(x(:, 3)-10).^2+4 * (x(:, 4)-5).^2+(x(:, 5)-3).^2+2 * (x(:, 6)-1).^2+5 * x(:, 7).^2+7 * (x(:, 8)-11).^2+2 * (x(:, 9)-10).^2+(x(:, 10)-7).^2+45;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG07Par()
            vari_num=10;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=-10*ones(1, 10);
            up_bou=10*ones(1, 10);
            x_best=[2.17199634142692,2.3636830416034,8.77392573913157,5.09598443745173,0.990654756560493,1.43057392853463,1.32164415364306,9.82872576524495,8.2800915887356,8.3759266477347];
            obj_best= 24.30620906818;
        end

        function [con,coneq]=singleG02Nonlcon(x)
            coneq=[];
            con(:, 1)=0.75-prod(x,2);
            con(:, 2)=sum(x,2)-7.5*10;
        end
        function obj=singleG02Obj(x)
            obj=-abs((sum(cos(x).^4,2)-2*prod(cos(x).^2,2))/sqrt(sum((1:10).*x.^2)));
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG02Par()
            vari_num=10;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=-0*ones(1, 10);
            up_bou=10*ones(1, 10);
            x_best=[2.17199634142692,2.3636830416034,8.77392573913157,5.09598443745173,0.990654756560493,1.43057392853463,1.32164415364306,9.82872576524495,8.2800915887356,8.3759266477347];
            obj_best= 24.30620906818;
        end

        % vari_num=13
        function obj=singleG01Obj(x)
            % vari_num=13;
            % obj_fcn=@(x) benchmark.singleG01Obj(x);
            % obj_fcn_low=@(x) benchmark.singleG01ObjLF(x);
            % Aineq=[
            %     2   2   0   0   0   0   0   0   0   1   1   0   0;
            %     2   0   2   0   0   0   0   0   0   1   0   1   0;
            %     0   2   2   0   0   0   0   0   0   0   1   1   0;
            %     -8  0   0   0   0   0   0   0   0   1   0   0   0;
            %     0   -8  0   0   0   0   0   0   0   0   1   0   0;
            %     0   0   -8  0   0   0   0   0   0   0   0   1   0
            %     0   0   0   -2  -1  0   0   0   0   1   0   0   0;
            %     0   0   0   0   0   -2  -1  0   0   0   1   0   0;
            %     0   0   0   0   0   0   0   -2  -1  0   0   1   0;
            %     ];
            % Bineq=[10;10;10;0;0;0;0;0;0];
            % Aeq=[];
            % Beq=[];
            % low_bou=zeros(1,13);
            % up_bou=ones(1,13);
            % up_bou(10:12)=100;
            % nonlcon_fcn=[];
            % nonlcon_fcn_LF=[];
            % model_function=@(x) objconFcn(x,@(x) benchmark.singleG01Obj(x),@(x) violationFunction(x,Aineq,Bineq,Aeq,Beq,[]));
            % cheapcon_function=[];
            %
            % x_best=[1,1,1,1,1,1,1,1,1,3,3,3,1], obj_best=-15;
            %
            sigma1=0;
            for i=1:4
                sigma1=sigma1+x(:,i);
            end
            sigma2=0;
            for i=1:4
                sigma2=sigma2+x(:,i).^2;
            end
            sigma3=0;
            for i=5:13
                sigma3=x(:,i)+sigma3;
            end
            obj=5*sigma1-5*sigma2-sigma3;
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG01Par()
            vari_num=13;
            Aineq=[
                2   2   0   0   0   0   0   0   0   1   1   0   0;
                2   0   2   0   0   0   0   0   0   1   0   1   0;
                0   2   2   0   0   0   0   0   0   0   1   1   0;
                -8  0   0   0   0   0   0   0   0   1   0   0   0;
                0   -8  0   0   0   0   0   0   0   0   1   0   0;
                0   0   -8  0   0   0   0   0   0   0   0   1   0
                0   0   0   -2  -1  0   0   0   0   1   0   0   0;
                0   0   0   0   0   -2  -1  0   0   0   1   0   0;
                0   0   0   0   0   0   0   -2  -1  0   0   1   0;
                ];
            Bineq=[10;10;10;0;0;0;0;0;0];
            Aeq=[];
            Beq=[];
            low_bou=[0,0,0,0,0,0,0,0,0,0,0,0,0];
            up_bou=[1,1,1,1,1,1,1,1,1,100,100,100,1];
            x_best=[1,1,1,1,1,1,1,1,1,3,3,3,1];
            obj_best=-15;
        end

        % vari_num=15
        function [con,coneq]=singleG19Nonlcon(x)
            popsize=size(x, 1);

            coneq=[];
            a=[-16 2 0 1 0;
                0 -2 0 0.4 2;
                -3.5 0 2 0 0;
                0 -2 0 -4 -1;
                0 -9 -2 1 -2.8;
                2 0 -4 0 0;
                -1 -1 -1 -1 -1;
                -1 -2 -3 -2 -1;
                1 2 3 4 5;
                1 1 1 1 1];
            b=[-40 -2 -0.25 -4 -4 -1 -40 -60 5 1];
            c=[30 -20 -10 32 -10;
                -20 39 -6 -31 32;
                -10 -6 10 -6 -10;
                32 -31 -6 39 -20;
                -10 32 -10 -20 30];
            d=[4 8 10 6 2];
            e=[-15 -27 -36 -18 -12];
            con(:, 1)=-2 * sum(repmat(c(1:5, 1)', popsize, 1).* x(:, 11:15), 2)-3 * d(1).* x(:, 11).^2-e(1)+sum(repmat(a(1:10, 1)', popsize, 1).* x(:, 1:10), 2);
            con(:, 2)=-2 * sum(repmat(c(1:5, 2)', popsize, 1).* x(:, 11:15), 2)-3 * d(2).* x(:, 12).^2-e(2)+sum(repmat(a(1:10, 2)', popsize, 1).* x(:, 1:10), 2);
            con(:, 3)=-2 * sum(repmat(c(1:5, 3)', popsize, 1).* x(:, 11:15), 2)-3 * d(3).* x(:, 13).^2-e(3)+sum(repmat(a(1:10, 3)', popsize, 1).* x(:, 1:10), 2);
            con(:, 4)=-2 * sum(repmat(c(1:5, 4)', popsize, 1).* x(:, 11:15), 2)-3 * d(4).* x(:, 14).^2-e(4)+sum(repmat(a(1:10, 4)', popsize, 1).* x(:, 1:10), 2);
            con(:, 5)=-2 * sum(repmat(c(1:5, 5)', popsize, 1).* x(:, 11:15), 2)-3 * d(5).* x(:, 15).^2-e(5)+sum(repmat(a(1:10, 5)', popsize, 1).* x(:, 1:10), 2);

        end
        function obj=singleG19Obj(x)
            popsize=size(x, 1);

            a=[-16 2 0 1 0;
                0 -2 0 0.4 2;
                -3.5 0 2 0 0;
                0 -2 0 -4 -1;
                0 -9 -2 1 -2.8;
                2 0 -4 0 0;
                -1 -1 -1 -1 -1;
                -1 -2 -3 -2 -1;
                1 2 3 4 5;
                1 1 1 1 1];
            b=[-40 -2 -0.25 -4 -4 -1 -40 -60 5 1];
            c=[30 -20 -10 32 -10;
                -20 39 -6 -31 32;
                -10 -6 10 -6 -10;
                32 -31 -6 39 -20;
                -10 32 -10 -20 30];
            d=[4 8 10 6 2];
            e=[-15 -27 -36 -18 -12];

            obj=sum(repmat(c(1:5, 1)', popsize, 1).* x(:, 11:15), 2).* x(:, 11)+sum(repmat(c(1:5, 2)', popsize, 1).* x(:, 11:15), 2).* x(:, 12)...
               +sum(repmat(c(1:5, 3)', popsize, 1).* x(:, 11:15), 2).* x(:, 13)+sum(repmat(c(1:5, 4)', popsize, 1).* x(:, 11:15), 2).* x(:, 14)...
               +sum(repmat(c(1:5, 5)', popsize, 1).* x(:, 11:15), 2).* x(:, 15)+2 * sum(repmat(d, popsize, 1).* x(:, 11:15).^3, 2)...
               -sum(repmat(b, popsize, 1).* x(:, 1:10), 2);
        end
        function [vari_num,Aineq,Bineq,Aeq,Beq,low_bou,up_bou,x_best,obj_best]=singleG19Par()
            vari_num=15;
            Aineq=[];
            Bineq=[];
            Aeq=[];
            Beq=[];
            low_bou=zeros(1, 15);
            up_bou=10*ones(1, 15);
            x_best=[0,0,3.9460,0,3.2832,10,0,0,0,0,0.3708,0.2785,0.5238,0.3886,0.2982];
            obj_best=32.6555929502463;
        end

    end
    
    methods(Static) % unconstraint multi Objective function
        function obj=multiZDT1Obj(x)
            vari_num=10;
            obj=zeros(size(x,1),2);
            obj(:,1)=x(:,1);
            g=1+9*(sum(x(:,2:vari_num),2)/(vari_num-1));
            obj(:,2)=g.*(1-sqrt(x(:,1)./g));
        end
        function obj=multiZDT2Obj(x)
            vari_num=10;
            obj=zeros(size(x,1),2);
            obj(:,1)=x(:,1);
            g=1+9*(sum(x(:,2:vari_num),2)/(vari_num-1));
            obj(:,2)=g.*(1-(x(:,1)./g)^2);
        end
        function obj=multiZDT3Obj(x)
            vari_num=10;
            obj=zeros(size(x,1),2);
            obj(:,1)=x(:,1);
            g=1+9*(sum(x(:,2:vari_num),2)/(vari_num-1));
            obj(:,2)=g.*(1-sqrt(x(:,1)./g)-(x(:,1)./g).*sin(10*pi*x(:,1)));
        end
    end
    
    methods(Static) % constraint multi Objective function
        function [con,coneq]=multiTNKNonlcon(x)
            con=zeros(2,1);
            x1=x(1);
            x2=x(2);
            if x1 == 0 && x2 == 0
                con(1)=-(x1^2+x2^2-1-0.1);
            else
                con(1)=-(x1^2+x2^2-1-0.1*cos(16*atan(x1/x2)));
            end
            con(2)=(x1-0.5)^2+(x2-0.5)^2-0.5;
            coneq=[];
        end
        function obj=multiTNKObj(x)
            % TNK problem
            % vari_num is 2
            %
            % obj_fcn=@(x) benchmark.multiTNKObj(x);
            % vari_num=2;
            % low_bou=zeros(1,2);
            % up_bou=ones(1,2)*pi;
            % nonlcon_fcn=@(x) multiTNKNonlcon(x);
            % cheapcon_function=[];
            %
            obj=zeros(2,1);
            obj(1)=x(1);
            obj(2)=x(2);
        end
    end
    
    methods(Static) % auxiliary function
        function [obj,con,coneq]=objconFcn(x,obj_fcn,nonlcon_fcn)
            % model function,concertrate obj,con,coneq into one function
            %
            if nargin < 3 || isempty(nonlcon_fcn)
                con=[];
                coneq=[];
            else
                [con,coneq]=nonlcon_fcn(x);
            end
            obj=obj_fcn(x);
        end

        function [con,coneq]=conFcn(x,Aineq,Bineq,Aeq,Beq,cheapcon_function)
            % convert Aineq, Bineq, Aeq, Beq to total cheapcon function
            % x input is rank vector
            %
            if nargin < 6
                cheapcon_function=[];
                if nargin < 5
                    Beq=[];
                    if nargin < 4
                        Aeq=[];
                        if nargin < 3
                            Bineq=[];
                            if nargin < 2
                                Aineq=[];
                            end
                        end
                    end
                end
            end
            con=[];
            coneq=[];
            if ~isempty(Aineq)
                if isempty(Bineq)
                    con=[con,x*Aineq'];
                else
                    con=[con,x*Aineq'-Bineq'];
                end
            end
            if ~isempty(Aeq)
                if isempty(Beq)
                    coneq=[coneq,x*Aeq'];
                else
                    coneq=[coneq,x*Aeq'-Beq'];
                end
            end
            if ~isempty(cheapcon_function)
                [lincon,linconeq]=cheapcon_function(x);
                con=[con,lincon];
                coneq=[coneq,linconeq];
            end
        end

        function [obj,con,coneq]=modelFcnLF(HF_model,x,error)
            [obj,con,coneq]=HF_model(x);
            obj=Benchmark.addError(x,obj,error(1,:));
            for con_idx=1:size(con,2)
                con(:,con_idx)=Benchmark.addError(x,con(:,con_idx),error(1+con_idx,:));
            end
            for coneq_idx=1:size(coneq,2)
                coneq(:,coneq_idx)=Benchmark.addError(x,coneq(:,coneq_idx),error(1+size(con,2)+coneq_idx,:));
            end
        end

        function value=addError(x,value,error)
            switch error(1)
                case 1
                    value=value+error(2)*Benchmark.biasE1(x,error(3));
                case 2
                    value=value+error(2)*Benchmark.biasE2(x,error(3));
                case 3
                    value=value+error(2)*Benchmark.biasE3(x,error(3));
                case 4
                    value=value+error(2)*Benchmark.biasE4(x,error(3));
            end
        end
    end

    methods(Static) % error bias function
        function bias=biasE1(x,fail)
            if nargin < 2
                fail=1000;
            end
            theta=1-0.0001*fail;
            bias=theta*sum(cos(10*pi*theta*x+0.5*pi*theta+pi),2);
        end

        function bias=biasE2(x,fail)
            if nargin < 2
                fail=1000;
            end
            theta=exp(-0.00025*fail);
            bias=theta*sum(cos(10*pi*theta*x+0.5*pi*theta+pi),2);
        end

        function bias=biasE3(x,fail)
            if nargin < 2
                fail=1000;
            end
            if fail < 1000
                theta=1-0.0002*fail;
            elseif fail < 2000
                theta=0.8;
            elseif fail < 3000
                theta=1.2-0.0002*fail;
            elseif fail < 4000
                theta=0.6;
            elseif fail < 5000
                theta=1.4-0.0002*fail;
            elseif fail < 6000
                theta=0.4;
            elseif fail < 7000
                theta=1.6-0.0002*fail;
            elseif fail < 8000
                theta=0.2;
            elseif fail < 9000
                theta=1.8-0.0002*fail;
            else
                theta=0;
            end

            bias=theta*sum(cos(10*pi*theta*x+0.5*pi*theta+pi),2);
        end

        function bias=biasE4(x,fail)
            if nargin < 2
                fail=1000;
            end
            theta=1-0.0001*fail;
            bias=theta*sum((1-abs(x)).*cos(10*pi*theta*x+0.5*pi*theta+pi),2);
        end
    end
end