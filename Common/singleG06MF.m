classdef singleG06MF < MultiFidelityBase
    properties
        name='G06MF';
        type='single';

        variable_number=2;
        low_bou = [13,0];
        up_bou = [100,100];
        x_best = [14.0950;0.8430]; 
        fval_best = -6.9618e+03;

        fval_number=1;
        con_number=2;
        coneq_number=0;
    end
    methods
        function [con,coneq] = calNonlconHF(self,x)
            g1 = -(x(:,1)-5).^2-(x(:,2)-5).^2+100;
            g2 = (x(:,1)-6).^2+(x(:,2)-5).^2-82.81;
            con = [g1,g2];
            coneq = [];
        end
        function fval = calObjHF(self,x)
            fval = (x(:,1)-10).^3+(x(:,2)-20).^3;
        end
        function [variable_number,low_bou,up_bou,x_best,fval_best] = getPar(self)
            variable_number = 2;
            low_bou = [13,0];
            up_bou = [100,100];
            x_best = [14.0950;0.8430];
            fval_best = -6.9618e+03;
        end
    end
    methods
        function [con,coneq] = calNonlconLF(self,x)
            [con,coneq] = self.calNonlconHF(x);
            con(:,1) = con(:,1)+2*self.biasE2(x);
            con(:,2) = con(:,2)+2*self.biasE3(x);
        end
        function fval = calObjLF(self,x)
            fval = self.calObjHF(x)-100*self.biasE1(x);
        end
    end
    methods
        function [fval,con,coneq]=calModelHF(self,x)
            fval=self.calObjHF(x);
            [con,coneq]=self.calNonlconHF(x);
        end
        function [fval,con,coneq]=calModelLF(self,x)
            fval=self.calObjLF(x);
            [con,coneq]=self.calNonlconLF(x);
        end
    end
end