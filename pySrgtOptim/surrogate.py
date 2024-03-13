import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class srgtKRG():
    '''
    generate Kriging surrogate model
    input data will be normalize by average and standard deviation of data

    input:
    X (matrix): x_num x vari_num matrix
    Y (matrix): x_num x 1 matrix
    model_option (struct): optional input

    model_option include:
    optimize_hyp, simplify_hyp, optimize_option, hyp, reg_fcn

    output:
    model_KRG (struct): a Kriging model

    abbreviation:
    num: number, pred: predict, vari: variable, hyp: hyper parameter
    NLL: negative log likelihood

    Copyright 2023.2 Adel
    '''

    def __init__(self, X, Y, model_option: dict = dict()):
        self.X = np.array(X)
        self.Y = np.array(Y)

        # Kriging option
        if 'optimize_hyp' not in model_option:
            model_option['optimize_hyp'] = True
        if 'simplify_hyp' not in model_option:
            model_option['simplify_hyp'] = True
        if model_option['simplify_hyp']:
            model_option['FLAG_GRAD'] = False
        else:
            model_option['FLAG_GRAD'] = True

        if 'optimize_option' not in model_option:
            model_option['optimize_option'] = {
                'disp': 'none', 'OptimalityTolerance': 1e-06, 'FiniteDifferenceStepSize': 1e-05, 'MaxIterations': 20}

        if 'hyp' not in model_option:
            model_option['hyp'] = []
        if 'reg_fcn' not in model_option:
            model_option['reg_fcn'] = None
        if 'cov_fcn' not in model_option:
            model_option['cov_fcn'] = None

        self.model_option = model_option

        # normalize data
        self.x_num, self.vari_num = self.X.shape
        self.Y = np.array(Y).reshape(self.x_num, 1)

        self.aver_X = np.mean(X, axis=0).reshape((1, self.vari_num))
        self.stdD_X = np.std(X, ddof=1, axis=0).reshape((1, self.vari_num))
        self.stdD_X[self.stdD_X == 0] = 1

        self.aver_Y = np.mean(Y, axis=0).reshape((1, 1))
        self.stdD_Y = np.std(Y, ddof=1, axis=0).reshape((1, 1))
        self.stdD_Y[self.stdD_Y == 0] = 1

        self.X_nomlz = (self.X - self.aver_X) / self.stdD_X
        self.Y_nomlz = (self.Y - self.aver_Y) / self.stdD_Y

        # covarianve function define
        cov_fcn = model_option['cov_fcn']
        if cov_fcn is None:
            # initial X_dis_sq
            X_dis_sq = np.zeros((self.x_num, self.x_num, self.vari_num))
            for vari_idx in range(0, self.vari_num):
                X_dis_sq[:, :, vari_idx] = (self.X_nomlz[:, vari_idx].reshape(
                    (-1, 1)) - self.X_nomlz[:, vari_idx].reshape((1, -1))) ** 2

            def covExp(X, X_pred, hyp, aver_X, stdD_X, X_nomlz, X_dis_sq):
                x_n, vari_n = X.shape
                theta = np.exp(hyp)
                if len(X_pred) == 0:
                    # calculate covariance
                    cov = np.zeros((x_n, x_n))
                    for vari_i in range(0, vari_n):
                        cov = cov + X_dis_sq[:, :, vari_i] * theta[vari_i]
                    cov = np.exp(- cov / vari_n ** 2) + np.eye(x_n) * \
                        ((1000 + x_n) * np.finfo(np.float64).eps)
                else:
                    x_pred_num = X_pred.shape[0]
                    X_pred_nomlz = (X_pred - aver_X) / stdD_X

                    # predict covariance
                    cov = np.zeros((x_n, x_pred_num))
                    for vari_i in range(0, vari_n):
                        cov = cov + (X_nomlz[:, vari_i].reshape(-1, 1) - X_pred_nomlz[:,
                                     vari_i].reshape((1, -1))) ** 2 * theta[vari_i]
                    cov = np.exp(- cov / vari_n ** 2)

                return cov

            def cov_fcn(X, X_pred, hyp): return covExp(
                X, X_pred, hyp, self.aver_X, self.stdD_X, self.X_nomlz, X_dis_sq)

        self.cov_fcn = cov_fcn

        # regression function define
        reg_fcn = model_option['reg_fcn']
        if reg_fcn is None:
            if self.x_num < self.vari_num:
                def reg_fcn(X=None): return np.ones(
                    (X.shape[1-1], 1))*self.stdD_Y + self.aver_Y
            else:
                def reg_fcn(X=None): return np.concatenate(
                    (np.ones((X.shape[0], 1)), X - self.stdD_X), axis=1)*self.stdD_Y + self.aver_Y
        self.reg_fcn = reg_fcn

    def train(self):
        # calculate reg
        fval_reg = self.reg_fcn(self.X)
        fval_reg_nomlz = (fval_reg - self.aver_Y) / self.stdD_Y

        hyp = self.model_option['hyp']
        # kernal function is exp(-X_sq/vari_num^2*exp(hyp))
        if len(hyp) == 0:
            hyp = np.ones((self.vari_num))

        # if isempty(hyp), hyp=log(x_num^(1/vari_num)*vari_num)*ones(1,vari_num);end

        # if optimize hyperparameter
        if self.model_option['optimize_hyp']:
            simplify_hyp = self.model_option['simplify_hyp']

            def obj_fcn_hyp(hyp): return self.probNLLKRG(hyp, fval_reg_nomlz)
            if simplify_hyp:
                hyp = np.mean(hyp)
                low_bou_hyp = - 4
                up_bou_hyp = 4
            else:
                low_bou_hyp = - 4 * np.ones((self.vari_num))
                up_bou_hyp = 4 * np.ones((self.vari_num))

            # [fval,gradient]=obj_fcn_hyp(hyp)
            # [~,gradient_differ]=differ(obj_fcn_hyp,hyp)
            # drawFcn(obj_fcn_hyp,low_bou_hyp,up_bou_hyp);

            hyp = minimize(obj_fcn_hyp, hyp, method='SLSQP', jac=None, bounds=Bounds(low_bou_hyp, up_bou_hyp),
                           options={'disp': False})
            hyp = hyp.x

            if simplify_hyp:
                hyp = hyp * np.ones((self.vari_num))

        # get parameter
        cov = self.cov_fcn(self.X_nomlz, [], hyp)
        L_cov, beta, sigma_sq, inv_L_F_reg, __, inv_L_U = self.calKRG(
            cov, self.Y_nomlz, self.x_num, fval_reg_nomlz)
        sigma_sq = sigma_sq * self.stdD_Y ** 2

        gamma = np.linalg.solve(np.transpose(L_cov), inv_L_U)
        inv_FTcovF = np.linalg.solve(
            (np.transpose(inv_L_F_reg) @ inv_L_F_reg), np.eye(fval_reg_nomlz.shape[2-1]))

        self.hyp = hyp
        self.L_cov = L_cov
        self.beta = beta
        self.inv_L_F_reg = inv_L_F_reg
        self.sigma_sq = sigma_sq
        self.gamma = gamma
        self.inv_FTcovF = inv_FTcovF

    def probNLLKRG(self, hyp, F_reg):
        # function to minimize sigma_sq

        if self.model_option['simplify_hyp']:
            hyp = hyp * np.ones((self.vari_num))

        R = self.cov_fcn(self.X_nomlz, [], hyp)

        L, Beta, sigma2, inv_L_F, __, __ = self.calKRG(
            R, self.Y_nomlz, self.x_num, F_reg)
        # calculation negative log likelihood
        if sigma2 == 0:
            fval = 0
            if self.model_option['simplify_hyp']:
                gradient = 0
            return fval

        fval = self.x_num / 2 * np.log(sigma2) + sum(np.log(np.diagonal(L)))

        return fval

    def calKRG(self, cov, Y, x_num, F_reg):
        # kriging interpolation kernel function
        # Y(x)=beta+Z(x)

        L_cov = np.linalg.cholesky(cov)

        inv_L_F_reg = np.linalg.solve(L_cov, F_reg)
        inv_L_Y = np.linalg.solve(L_cov, Y)

        # basical bias
        beta = np.linalg.solve(np.transpose(inv_L_F_reg) @
                               inv_L_F_reg, np.transpose(inv_L_F_reg) @ inv_L_Y)
        inv_L_U = inv_L_Y - inv_L_F_reg @ beta
        sigma_sq = np.sum(inv_L_U ** 2) / x_num

        return L_cov, beta, sigma_sq, inv_L_F_reg, inv_L_Y, inv_L_U

    def predict(self, X_pred):
        '''
        Kriging surrogate predict function

        input:
        X_pred (matrix): x_pred_num x vari_num matrix, predict X

        output:
        Y_pred (matrix): x_pred_num x 1 matrix, value
        Var_pred (matrix): x_pred_num x 1 matrix, variance
        '''

        X_pred = np.array(X_pred).reshape((-1, self.vari_num))
        fval_reg_pred_nomlz = (self.reg_fcn(X_pred) -
                               self.aver_Y) / self.stdD_Y
        cov_pred = self.cov_fcn(self.X, X_pred, self.hyp)

        # predict base fval
        Y_pred = fval_reg_pred_nomlz @ self.beta + \
            np.transpose(cov_pred) @ self.gamma
        # predict variance
        inv_L_r = np.linalg.solve(self.L_cov, cov_pred)
        u = np.transpose((self.inv_L_F_reg)) @ inv_L_r - \
            np.transpose(fval_reg_pred_nomlz)
        Var_pred = self.sigma_sq * (1 + np.transpose(u) @
                                    self.inv_FTcovF @ u - np.transpose(inv_L_r) @ inv_L_r)
        Var_pred = np.diagonal(Var_pred)
        # renormalize data
        Y_pred = Y_pred * self.stdD_Y + self.aver_Y

        return Y_pred, Var_pred


class srgtRBF():
    '''
    radial basis fcn surrogate pre model fcn
    input initial data X,Y,which are real data
    X,Y are x_num x vari_num matrix
    aver_X,stdD_X is 1 x x_num matrix
    output is a radial basis model,include X,Y,base_fcn
    and predict_fcn

    Copyright 2023.2 Adel
    '''

    def __init__(self, X, Y, basis_fcn=lambda r: r**3):
        self.X = np.array(X)
        self.basis_fcn = basis_fcn

        self.x_num, self.vari_num = self.X.shape
        self.Y = np.array(Y).reshape(self.x_num, 1)

        self.aver_X = np.mean(X, axis=0).reshape((1, self.vari_num))
        self.stdD_X = np.std(X, ddof=1, axis=0).reshape((1, self.vari_num))
        self.stdD_X[self.stdD_X == 0] = 1

        self.aver_Y = np.mean(Y, axis=0).reshape((1, 1))
        self.stdD_Y = np.std(Y, ddof=1, axis=0).reshape((1, 1))
        self.stdD_Y[self.stdD_Y == 0] = 1

    def train(self):
        x_num = self.x_num
        vari_num = self.vari_num

        # normalize data
        self.X_nomlz = (self.X - self.aver_X) / self.stdD_X
        self.Y_nomlz = (self.Y - self.aver_Y) / self.stdD_Y

        # initialization distance of all X
        X_dis = np.zeros((x_num, x_num))
        for vari_idx in range(vari_num):
            X_dis = X_dis + (self.X_nomlz[:, vari_idx].reshape(x_num, 1) -
                             np.transpose(self.X_nomlz[:, vari_idx]).reshape(1, x_num)) ** 2

        X_dis = np.sqrt(X_dis)

        self.rdibas_matrix = self.basis_fcn(X_dis)
        # stabilize matrix
        self.rdibas_matrix = self.rdibas_matrix + np.eye(self.x_num) * 1e-09
        # get inverse matrix
        self.inv_rdibas_matrix = np.linalg.solve(
            self.rdibas_matrix, np.eye(self.x_num))
        # solve beta
        self.beta = np.dot(self.inv_rdibas_matrix, self.Y_nomlz)

    def predict(self, X_pred):
        X_pred = np.array(X_pred).reshape((-1, self.vari_num))

        x_pred_num, __ = X_pred.shape
        # normalize data
        X_pred_nomlz = (X_pred - self.aver_X) / self.stdD_X
        # calculate distance
        X_dis_pred = np.zeros((x_pred_num, self.x_num))
        for vari_idx in range(self.vari_num):
            X_dis_pred = X_dis_pred + \
                (X_pred_nomlz[:, vari_idx].reshape(x_pred_num, 1) -
                 np.transpose(self.X_nomlz[:, vari_idx]).reshape(1, self.x_num)) ** 2

        X_dis_pred = np.sqrt(X_dis_pred)
        # predict variance
        Y_pred_nomlz = np.dot(self.basis_fcn(X_dis_pred), self.beta)
        # normalize data
        Y_pred = Y_pred_nomlz * self.stdD_Y + self.aver_Y

        return Y_pred


class srgtRBFMF():
    '''
    radial basis fcn surrogate pre model fcn
    input initial data X,Y,which are real data
    X,Y are x_num x vari_num matrix
    aver_X,stdD_X is 1 x x_num matrix
    output is a radial basis model,include X,Y,base_fcn
    and predict_fcn

    Copyright 2023.2 Adel
    '''

    def __init__(self, X_HF, Y_HF, basis_fcn_HF=lambda r: r**3, X_LF=None, Y_LF=None, basis_fcn_LF=lambda r: r**3):
        self.X_HF = np.array(X_HF)
        self.basis_fcn_HF = basis_fcn_HF
        self.x_HF_num, self.vari_num = self.X_HF.shape
        self.Y_HF = np.array(Y_HF).reshape(self.x_HF_num, 1)

        self.X_LF = np.array(X_LF)
        self.basis_fcn_LF = basis_fcn_LF
        self.Y_LF = np.array(Y_LF).reshape(self.X_LF.shape[0], 1)

        self.aver_X = np.mean(X_HF, axis=0).reshape((1, self.vari_num))
        self.stdD_X = np.std(X_HF, ddof=1, axis=0).reshape((1, self.vari_num))
        self.stdD_X[self.stdD_X == 0] = 1

        self.aver_Y = np.mean(Y_HF, axis=0).reshape((1, 1))
        self.stdD_Y = np.std(Y_HF, ddof=1, axis=0).reshape((1, 1))
        self.stdD_Y[self.stdD_Y == 0] = 1

    def train(self):
        # train LF
        LF_model = srgtRBF(self.X_LF, self.Y_LF, self.basis_fcn_LF)
        LF_model.train()
        self.LF_model = LF_model

        x_HF_num = self.x_HF_num
        vari_num = self.vari_num

        # normalize data
        self.X_nomlz = (self.X_HF - self.aver_X) / self.stdD_X
        self.Y_nomlz = (self.Y_HF - self.aver_Y) / self.stdD_Y
        YHF_pred = LF_model.predict(self.X_HF)
        YHF_pred_nomlz = (YHF_pred - self.aver_Y) / self.stdD_Y

        # initialization distance of all X
        X_dis = np.zeros((x_HF_num, x_HF_num))
        for vari_idx in range(vari_num):
            X_dis = X_dis + (self.X_nomlz[:, vari_idx].reshape(x_HF_num, 1) -
                             np.transpose(self.X_nomlz[:, vari_idx]).reshape(1, x_HF_num)) ** 2

        X_dis = np.sqrt(X_dis)

        self.rdibas_matrix = self.basis_fcn_HF(X_dis)
        # add low fildelity value
        self.H = np.concatenate(
            (self.rdibas_matrix*YHF_pred_nomlz, self.rdibas_matrix), axis=1)
        self.H_hessian = np.dot(self.H, np.transpose(self.H))
        # stabilize matrix
        self.H_hessian = self.H_hessian + np.eye(x_HF_num) * 1e-09
        # get inv matrix
        self.inv_H_hessian = np.linalg.solve(self.H_hessian, np.eye(x_HF_num))
        # solve omega
        self.omega = np.dot(np.transpose(self.H), np.dot(
            self.inv_H_hessian, self.Y_nomlz))
        self.alpha = self.omega[:x_HF_num]
        self.beta = self.omega[x_HF_num:]

    def predict(self, X_pred):
        X_pred = np.array(X_pred).reshape((-1, self.vari_num))

        x_pred_num, __ = X_pred.shape
        # normalize data
        X_pred_nomlz = (X_pred - self.aver_X) / self.stdD_X
        # calculate distance
        X_dis_pred = np.zeros((x_pred_num, self.x_HF_num))
        for vari_idx in range(self.vari_num):
            X_dis_pred = X_dis_pred + \
                (X_pred_nomlz[:, vari_idx].reshape(x_pred_num, 1) -
                 np.transpose(self.X_nomlz[:, vari_idx]).reshape(1, self.x_HF_num)) ** 2

        X_dis_pred = np.sqrt(X_dis_pred)

        # predict low fildelity value
        Y_pred_LF = self.LF_model.predict(X_pred)
        # nomalizae
        Y_pred_LF_nomlz = (Y_pred_LF - self.aver_Y) / self.stdD_Y

        # combine two matrix
        rdibas_matrix_pred = self.basis_fcn_HF(X_dis_pred)
        H_pred = np.concatenate(
            (rdibas_matrix_pred*Y_pred_LF_nomlz, rdibas_matrix_pred), axis=1)
        # predict data
        Y_pred_nomlz = np.dot(H_pred, self.omega)
        # normalize data
        Y_pred = Y_pred_nomlz * self.stdD_Y + self.aver_Y

        return Y_pred


def surrogateVisualize(mdl, low_bou, up_bou):
    '''
    visualize surrogate model
    '''
    draw_X, draw_Y = np.meshgrid(np.linspace(
        low_bou[0], up_bou[0], 21), np.linspace(low_bou[1], up_bou[1], 21))
    draw_Point = np.concatenate(
        (draw_X.reshape((441, 1)), draw_Y.reshape((441, 1))), axis=1)
    draw_Z, __ = mdl.predict(draw_Point)
    draw_Z = draw_Z.reshape((21, 21))

    fig = plt.figure()
    axe = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(axe)
    axe.scatter(mdl.X[:, 0], mdl.X[:, 1], mdl.Y[:, 0], s=100, color='red', marker='o')
    axe.plot_surface(draw_X, draw_Y, draw_Z, alpha=0.5)

    axe.set_xlabel(r'$x$', fontsize=18)
    axe.set_ylabel(r'$y$', fontsize=18)
    axe.set_zlabel(r'$z$', fontsize=18)
    axe.view_init(30, -120)
    plt.axis('auto')
    plt.show()


if __name__ == '__main__':
    from scipy import io

    # data = io.loadmat('Forrester.mat')

    # srgt = srgtRBFMF(data['XHF'], data['YHF'],X_LF=data['XLF'],Y_LF=data['YLF'])
    # srgt.train()

    # draw_X=np.linspace(0,1,20).reshape(20,1)
    # draw_Y = srgt.predict(draw_X)

    # plt.plot(draw_X,draw_Y)
    # plt.plot(data['x'],data['y_real'],'-')
    # plt.plot(data['x'],data['y_real_low'],'--')
    # plt.axis('auto')
    # plt.show()

    data = io.loadmat('surrogate/PK.mat')
    low_bou = data['low_bou'].reshape(2,)
    up_bou = data['up_bou'].reshape(2,)

    # srgt = srgtRBF(data['X'], data['Y'])
    srgt = srgtKRG(data['X'], data['Y'], {'simplify_hyp': False})

    srgt.train()

    surrogateVisualize(srgt, low_bou, up_bou)
