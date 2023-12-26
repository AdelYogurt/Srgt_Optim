classdef singleG07MF < MultiFidelityBase
    properties
        name='G07MF';
        type='single';

        variable_number=10;
        low_bou = -10*ones(1, 10);
        up_bou = 10*ones(1, 10);
        x_best = [2.17199634142692,2.3636830416034,8.77392573913157,5.09598443745173,0.990654756560493,1.43057392853463,1.32164415364306,9.82872576524495,8.2800915887356,8.3759266477347];
        fval_best =  24.30620906818;

        fval_number=1;
        con_number=8;
        coneq_number=0;
    end
    methods
        function [con,coneq] = calNonlconHF(self,x)
            coneq = [];
            con(:, 1) = -105 + 4 * x(:, 1) + 5 * x(:, 2) - 3 * x(:, 7) + 9 * x(:, 8);
            con(:, 2) = 10 * x(:, 1) - 8 * x(:, 2) - 17 * x(:, 7) + 2 * x(:, 8);
            con(:, 3) = -8 * x(:, 1) + 2 * x(:, 2) + 5 * x(:, 9) - 2 * x(:, 10) - 12;
            con(:, 4) = 3 * (x(:, 1) - 2).^2 + 4 * (x(:, 2) - 3).^2 + 2 * x(:, 3).^2 - 7 * x(:, 4) - 120;
            con(:, 5) = 5 * x(:, 1).^2 + 8 * x(:, 2) + (x(:, 3) - 6).^2 - 2 * x(:, 4) - 40;
            con(:, 6) = x(:, 1).^2 + 2 * (x(:, 2) - 2).^2 - 2 * x(:, 1) .* x(:, 2) + 14 * x(:, 5) - 6 * x(:, 6);
            con(:, 7) = 0.5 * (x(:, 1) - 8).^2 + 2 * (x(:, 2) - 4).^2 + 3 * x(:, 5).^2 - x(:, 6) - 30;
            con(:, 8) = -3 * x(:, 1) + 6 * x(:, 2) + 12 * (x(:, 9) - 8).^2 - 7 * x(:, 10);
        end
        function fval = calObjHF(self,x)
            fval = x(:, 1).^2 + x(:, 2).^2 + x(:, 1) .* x(:, 2) - 14 * x(:, 1) - 16 * x(:, 2) + (x(:, 3) - 10).^2 + 4 * (x(:, 4) - 5).^2 + (x(:, 5) - 3).^2 + 2 * (x(:, 6) - 1).^2 + 5 * x(:, 7).^2 + 7 * (x(:, 8) - 11).^2 + 2 * (x(:, 9) - 10).^2 + (x(:, 10) - 7).^2 + 45;
        end
        function [variable_number,low_bou,up_bou,x_best,fval_best] = getPar(self)
            variable_number=10;
            low_bou = -10*ones(1, 10);
            up_bou = 10*ones(1, 10);
            x_best = [2.17199634142692,2.3636830416034,8.77392573913157,5.09598443745173,0.990654756560493,1.43057392853463,1.32164415364306,9.82872576524495,8.2800915887356,8.3759266477347];
            fval_best =  24.30620906818;
        end
    end
    methods
        function [con,coneq] = calNonlconLF(self,x)
            [con,coneq] = self.calNonlconHF(x);
            con(:,1) = con(:,1)+1*self.biasE2(x,1000);
            con(:,2) = con(:,2)+1*self.biasE3(x,1000);
            con(:,3) = con(:,3)+1*self.biasE4(x,1000);
            con(:,4) = con(:,4)+1*self.biasE1(x,1000);
            con(:,5) = con(:,5)+1*self.biasE2(x,1000);
            con(:,6) = con(:,6)+1*self.biasE3(x,1000);
            con(:,7) = con(:,7)+1*self.biasE4(x,1000);
            con(:,8) = con(:,8)+1*self.biasE1(x,1000);
        end
        function fval = calObjLF(self,x)
            fval = self.calObjHF(x)+10*self.biasE1(x,100);
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