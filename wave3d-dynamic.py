'''
Author: Chenfa Zheng
Last Edit: 2022.04.22
'''


import configparser
import tensorflow as tf
import numpy as np
from time import time
import os
import yaml
import imageio
import copy

import matplotlib.pyplot as plt


# set data typea=
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


class DataGenerator():
    def __init__(self, model, tmin, tmax, xmin, xmax, ymin, ymax, zmin, zmax, relative_noise=0.1):
        self.model = model
        self.tmin = tmin
        self.tmax = tmax
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax
        self.nr = relative_noise
    
    def _gen_relative_noise(self, n):
        return tf.random.normal((n, 1), dtype=DTYPE) * self.nr
    
    def get_lb_ub(self):
        lb = tf.constant([self.tmin, self.xmin, self.ymin, self.zmin], dtype=DTYPE)
        ub = tf.constant([self.tmax, self.xmax, self.ymax, self.zmax], dtype=DTYPE)
        return lb, ub
    
    def gen_bound(self, N_bound):
        # x0 = tf.random.uniform((N_bound, 1), minval=self.xmin, maxval=self.xmax, dtype=DTYPE)
        # y0 = tf.random.uniform((N_bound, 1), minval=self.ymin, maxval=self.ymax, dtype=DTYPE)
        # z0 = tf.random.uniform((N_bound, 1), minval=self.zmin, maxval=self.zmax, dtype=DTYPE)
        x0 = tf.reshape(tf.linspace(self.xmin, self.xmax, N_bound), (N_bound, 1))
        y0 = tf.reshape(tf.linspace(self.ymin, self.ymax, N_bound), (N_bound, 1))
        z0 = tf.reshape(tf.linspace(self.zmin, self.zmax, N_bound), (N_bound, 1))
        xx0, yy0, zz0 = tf.meshgrid(x0, y0, z0)
        xxx0 = tf.reshape(xx0, (N_bound**3, 1))
        yyy0 = tf.reshape(yy0, (N_bound**3, 1))
        zzz0 = tf.reshape(zz0, (N_bound**3, 1))
        ttt0 = tf.ones_like(xxx0) * self.tmin
        X0 = tf.concat([ttt0, xxx0, yyy0, zzz0], 1)

        u0 = self.model.u0(X0) * (1.0 + self._gen_relative_noise(N_bound**3))
        ut0 = self.model.ut0(X0) * (1.0 + self._gen_relative_noise(N_bound**3))
        
        return X0, u0, ut0
    
    def gen_train(self, N_train):
        t_train = tf.random.uniform((N_train, 1), minval=self.tmin, maxval=self.tmax, dtype=DTYPE)
        x_train = tf.random.uniform((N_train, 1), minval=self.xmin, maxval=self.xmax, dtype=DTYPE)
        y_train = tf.random.uniform((N_train, 1), minval=self.ymin, maxval=self.ymax, dtype=DTYPE)
        z_train = tf.random.uniform((N_train, 1), minval=self.zmin, maxval=self.zmax, dtype=DTYPE)
        X_train = tf.concat([t_train, x_train, y_train, z_train], 1)
        return X_train
    
    def gen_predict(self, N_predict, t=None):
        x = tf.cast(tf.linspace(self.xmin, self.xmax, N_predict), dtype=DTYPE)
        y = tf.cast(tf.linspace(self.ymin, self.ymax, N_predict), dtype=DTYPE)
        z = tf.cast(tf.linspace(self.zmin, self.zmax, N_predict), dtype=DTYPE)
        xm, ym, zm = tf.meshgrid(x, y, z)
        xx = tf.reshape(xm, (N_predict**3, 1))
        yy = tf.reshape(ym, (N_predict**3, 1))
        zz = tf.reshape(zm, (N_predict**3, 1))
        if t is None:
            t_view = self.tmax
        else:
            t_view = t
        tt = tf.ones_like(xx) * t_view
        # tt = tf.random.uniform((N_predict**3, 1), minval=self.tmin, maxval=self.tmax, dtype=DTYPE)
        X = tf.concat([tt, xx, yy, zz], 1)
        return X


class DataModel():
    def __init__(self, A, ome, k1, k2, k3):
        self.A = A
        self.ome = ome
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
    
    def u_model(self, X):
        t = X[:, 0:1]
        x = X[:, 1:2]
        y = X[:, 2:3]
        z = X[:, 3:4]
        u = self.A * tf.sin(self.ome * t + self.k1 * x + self.k2 * y + self.k3 * z) + self.A
        return u
    
    def u0(self, X):
        return self.u_model(X)
    
    def ut0(self, X):
        with tf.GradientTape() as g:
            g.watch(X)
            u = self.u0(X)
        u_t = g.batch_jacobian(u, X)[:, :, 0]
        return u_t
    
    def get_a(self):
        a1 = self.ome / self.k1
        a2 = self.ome / self.k2
        a3 = self.ome / self.k3
        return a1, a2, a3


class Plotter():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_sample(self, X0, u0, ut0, X_train):
        t0, x0, y0, z0 = X0[:, 0:1], X0[:, 1:2], X0[:, 2:3], X0[:, 3:4]
        t, x, y, z = X_train[:, 0:1], X_train[:, 1:2], X_train[:, 2:3], X_train[:, 3:4]

        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x0, y0, z0, c='k', marker='s', s=6, alpha=0.5, label='boundary')
        cax = ax.scatter(x, y, z, c=t, cmap='viridis', marker='o', s=6, alpha=0.5, label='train')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.set_title('Positions of collocation points and boundary data')
        # ax.legend()
        ax.view_init(elev=20, azim=-120)
        fig.colorbar(cax, ax=ax, fraction=0.02, pad=0.1)
        plt.savefig('%s/Sample3D.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()
    
    def plot_bound(self, X0, u0, ut0):
        t0, x0, y0, z0 = X0[:, 0:1], X0[:, 1:2], X0[:, 2:3], X0[:, 3:4]

        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(x0, y0, z0, c=u0, alpha=0.5, marker='.', cmap='coolwarm', label='boundary')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.view_init(elev=27, azim=158)
        ax.set_title('Boundary data')
        plt.savefig('%s/InitBound3D.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

    def plot_bound_surface(self, X0, u0, N, title='Boundary'):
        xx = tf.reshape(X0[:, 1:2], (N, N, N))
        yy = tf.reshape(X0[:, 2:3], (N, N, N))
        zz = tf.reshape(X0[:, 3:4], (N, N, N))
        uu = tf.reshape(u0, (N, N, N))
        xmin, xmax = tf.reduce_min(xx), tf.reduce_max(xx)
        ymin, ymax = tf.reduce_min(yy), tf.reduce_max(yy)
        zmin, zmax = tf.reduce_min(zz), tf.reduce_max(zz)
        umin, umax = tf.reduce_min(uu), tf.reduce_max(uu)

        fig = plt.figure(figsize=(9,6))
        kws = {
            'vmin': umin, 
            'vmax': umax,
            'levels': np.linspace(umin, umax, 100),
            'cmap': 'cividis',
        }
        ax = fig.add_subplot(111, projection='3d')
        _ = ax.contourf(
            xx[:, :, -1], yy[:, :, -1], uu[:, :, -1],
            zdir='z', offset=zmax, **kws
        )
        _ = ax.contourf(
            xx[0, :, :], uu[0, :, :], zz[0, :, :], 
            zdir='y', offset=ymin, **kws
        )
        C = ax.contourf(
            uu[:, 0, :], yy[:, 0, :], zz[:, 0, :],
            zdir='x', offset=xmin, **kws
        )
        ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))

        # plot edges
        edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
        ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
        ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
        ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$z$')
        ax.set(
            xticks = np.linspace(xmin, xmax, 5),
            yticks = np.linspace(ymin, ymax, 5),
            zticks = np.linspace(zmin, zmax, 5),
        )
        ax.view_init(elev=20, azim=-120)
        fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1)
        ax.set_title(title)
        plt.savefig('%s/%s3D.jpg'%(self.save_dir, title), bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()
    
    def plot_surface_animated(self, ts, Xs, us, N, title='Predict'):
        umin = min([tf.reduce_min(u) for u in us])
        umax = max([tf.reduce_max(u) for u in us])
        kws = {
            'vmin': umin, 
            'vmax': umax,
            'levels': np.linspace(umin, umax, 100),
            'cmap': 'cividis',
        }
        imgs = []
        for i in range(len(Xs)):
            fig = plt.figure(figsize=(9,6))
            ax = fig.add_subplot(111, projection='3d')

            t0 = ts[i]
            X0 = Xs[i]
            u0 = us[i]
            xx = tf.reshape(X0[:, 1:2], (N, N, N))
            yy = tf.reshape(X0[:, 2:3], (N, N, N))
            zz = tf.reshape(X0[:, 3:4], (N, N, N))
            uu = tf.reshape(u0, (N, N, N))
            xmin, xmax = tf.reduce_min(xx), tf.reduce_max(xx)
            ymin, ymax = tf.reduce_min(yy), tf.reduce_max(yy)
            zmin, zmax = tf.reduce_min(zz), tf.reduce_max(zz)
            
            _ = ax.contourf(
                xx[:, :, -1], yy[:, :, -1], uu[:, :, -1],
                zdir='z', offset=zmax, **kws
            )
            _ = ax.contourf(
                xx[0, :, :], uu[0, :, :], zz[0, :, :], 
                zdir='y', offset=ymin, **kws
            )
            C = ax.contourf(
                uu[:, 0, :], yy[:, 0, :], zz[:, 0, :],
                zdir='x', offset=xmin, **kws
            )
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), zlim=(zmin, zmax))

            # plot edges
            edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
            ax.plot([xmin, xmin], [ymin, ymax], [zmax, zmax], **edges_kw)
            ax.plot([xmin, xmax], [ymin, ymin], [zmax, zmax], **edges_kw)
            ax.plot([xmin, xmin], [ymin, ymin], [zmin, zmax], **edges_kw)

            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')
            ax.set(
                xticks = np.linspace(xmin, xmax, 5),
                yticks = np.linspace(ymin, ymax, 5),
                zticks = np.linspace(zmin, zmax, 5),
            )
            ax.view_init(elev=20, azim=-120)
            fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1)
            ax.set_title(title + ' at t = %.2f'%t0)

            img_path = '%s/%s_%d.png'%(self.save_dir, title, i)
            plt.savefig(img_path, bbox_inches='tight', dpi=300)
            plt.close()
            imgs.append(img_path)
        
        # make gif
        gif_imgs = [imageio.imread(img) for img in imgs]
        imageio.mimsave(os.path.join(self.save_dir, '%s.gif'%title), gif_imgs, 'GIF', duration=0.1)

    
    def plot_predict_3d(self, tt, xx, U_pred, U_true, show=False, view_init=(35, 35)):
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(tt, xx, U_pred, cmap='viridis')
        ax.plot_surface(tt, xx, U_true, cmap='viridis', alpha=0.15)
        ax.view_init(*view_init)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_zlabel('$u_\\theta(t,x)$')
        ax.set_title('Solution of wave equation')
        plt.savefig('%s/Solution3D.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()
    
    def plot_predict_2d(self, tt, xx, U_pred, show=False):
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        ax.contour(tt, xx, U_pred, colors='black', alpha=0.5)
        sc = ax.scatter(tt, xx, c=U_pred, cmap='viridis', s=1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Solution of wave equation')
        plt.colorbar(sc)
        plt.savefig('%s/Solution2D.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()

    def plot_real_2d(self, tt, xx, U_true, show=False):
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        ax.contour(tt, xx, U_true, colors='black', alpha=0.5)
        sc = ax.scatter(tt, xx, c=U_true, cmap='viridis', s=1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('True value of wave equation')
        plt.colorbar(sc)
        plt.savefig('%s/True2D.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()
    
    def plot_resudials_2d(self, tt, xx, U_pred, U_true, show=False):
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        ax.contour(tt, xx, U_pred - U_true, colors='black', alpha=0.5)
        sc = ax.scatter(tt, xx, c=U_pred-U_true, cmap='viridis', s=1)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Residuals of wave equation')
        plt.colorbar(sc)
        plt.savefig('%s/Resudials2D.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()
    
    def plot_loss_history(self, history, show=False, title='LossTrain'):
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111)
        ax.semilogy(range(len(history)), history,'k-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        plt.savefig('%s/%s.jpg'%(self.save_dir, title), bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()


class WaveModel(tf.keras.Sequential):
    def __init__(self, layers, lb, ub, a1, a2, a3):
        super().__init__()
        # 2 dimensional input, t and x
        self.add(tf.keras.Input(4))
        # scaling layer
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0 * (x - lb) / (ub - lb) - 1.0
        )
        self.add(scaling_layer)
        # hidden layers
        for n in layers:
            self.add(tf.keras.layers.Dense(
                n, 
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer=tf.keras.initializers.GlorotNormal()
            ))
        # output layer
        self.add(tf.keras.layers.Dense(1))

        # additional parameters
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
    
    def predict(self, X):
        return self.call(X)

    # residual of wave equation
    def _func_r(self, u_tt, u_xx, u_yy, u_zz):
        return u_tt - self.a1 * u_xx - self.a2 * u_yy - self.a3 * u_zz
    
    # get residual of wave equation
    def get_r(self, X):
        with tf.GradientTape(persistent=True) as tape:
            t, x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 3:4]
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)
            tape.watch(z)

            u = self.call(tf.stack([t[:, 0], x[:, 0], y[:, 0], z[:, 0]], axis=1))
            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)
            u_z = tape.gradient(u, z)

        u_tt = tape.gradient(u_t, t)
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)
        u_zz = tape.gradient(u_z, z)

        r = self._func_r(u_tt, u_xx, u_yy, u_zz)
        return r
    
    def get_bound_r(self, X0, u0, ut0):
        # boundary loss at u(x, t=0)
        u0_pred = self.call(X0)
        l1 = tf.reduce_mean(tf.square(u0_pred - u0))

        # boundary loss at u_t(x, t=0)
        with tf.GradientTape() as tape:
            tape.watch(X0)
            u0_pred = self.call(X0)
        ut0_pred = tape.batch_jacobian(u0_pred, X0)[:, :, 0]
        del tape
        l2 = tf.reduce_mean(tf.square(ut0_pred - ut0))

        return l1, l2
    
    def compute_loss(self, X0, u0, ut0, X_train, X0_val, u0_val, ut0_val):
        # equation loss
        r = self.get_r(X_train)
        l0 = tf.reduce_mean(tf.square(r))

        # boundary loss at u(x, t=0)
        u0_pred = self.call(X0)
        l1 = tf.reduce_mean(tf.square(u0_pred - u0))

        u0_pred_val = self.call(X0_val)
        l1_val = tf.reduce_mean(tf.square(u0_pred_val - u0_val))

        # boundary loss at u_t(x, t=0)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(X0)
            tape.watch(X0_val)
            u0_pred = self.call(X0) 
            u0_pred_val = self.call(X0_val)
        ut0_pred = tape.batch_jacobian(u0_pred, X0)[:, :, 0]
        ut0_pred_val = tape.batch_jacobian(u0_pred_val, X0_val)[:, :, 0]
        l2 = tf.reduce_mean(tf.square(ut0_pred - ut0))
        l2_val = tf.reduce_mean(tf.square(ut0_pred_val - ut0_val))

        # loss summary
        return l0, l1, l2, l1_val, l2_val
    
    def get_grad(self, X0, u0, ut0, X_train, w_u0, w_ut0, X0_val, u0_val, ut0_val, ctr):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            l0, l1, l2, l1_val, l2_val = self.compute_loss(X0, u0, ut0, X_train, X0_val, u0_val, ut0_val)
            if ctr == 0:
                m = (1 + w_u0**2 + w_ut0**2)**(-0.5)
                w1, w2, w3 = m, w_u0*m, w_ut0*m
                print('ctr: w1, w2, w3: ', ctr, w1, w2, w3, flush=True)
            else:
                last_loss = self.loss_train[-1]
                dl1, dl2, dl3 = abs(l0 - last_loss[0])+1e-12, abs(l1 - last_loss[1])+1e-12, abs(l2 - last_loss[2])+1e-12
                k = (1/dl1**2 + 1/dl2**2 + 1/dl3**2)**(-0.5)
                w1, w2, w3 = k/dl1, k/dl2, k/dl3
                print('ctr: w1, w2, w3: ', ctr, w1, w2, w3, flush=True)
            loss = w1 * l0 + w2 * l1 + w3 * l2
        print('loss', loss, l0, l1, l2)
        self.loss_weight.append([w1, w2, w3])
        self.loss_train.append(float(loss))
        self.loss_val.append(float(l1_val + l2_val))
        grad = tape.gradient(loss, self.trainable_variables)
        return grad
    
    def fit(self, X0, u0, ut0, X_train, w_u0, w_ut0, N_iter=1000, optimizer=None):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # split data: 80% for train, 20% for validation
        X0_train, X0_val = tf.split(X0, [int(0.8*X0.shape[0]), int(0.2*X0.shape[0])], axis=0)
        u0_train, u0_val = tf.split(u0, [int(0.8*u0.shape[0]), int(0.2*u0.shape[0])], axis=0)
        ut0_train, ut0_val = tf.split(ut0, [int(0.8*ut0.shape[0]), int(0.2*ut0.shape[0])], axis=0)

        @tf.function
        def train_step(i):
            grad_theta = self.get_grad(X0_train, u0_train, ut0_train, X_train, w_u0, w_ut0, X0_val, u0_val, ut0_val, ctr=i)
            optimizer.apply_gradients(zip(grad_theta, self.trainable_variables))
            pass

        t0 = time()

        self.loss_train = []
        self.loss_val = []
        self.loss_weight = []

        for i in range(N_iter+1):
            train_step(i)

            if i%10 == 0:
                print(i, self.loss_train[-1], self.loss_val[-1], flush=True)
                print('iteration: {:05d}, train loss: {:10.8e} test bound loss: {:10.8e}'.format(i, self.loss_train[-1], self.loss_val[-1]))

        print('time: {:.2f} sec'.format(time() - t0))
        print('final train loss: {:10.8e}'.format(self.loss_train[-1]))
        
        pass

def run(config, work_dir):
    # set random seed
    np.random.seed(config['randseed'])
    tf.random.set_seed(config['randseed'])

    # create directory
    save_dir = os.path.join(work_dir, config['name'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig_dir = os.path.join(save_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # create plotter and set directory
    plotter = Plotter(fig_dir)


    # --------------------------------------------------
    # create training data
    # --------------------------------------------------

    A = config['data']['A']
    ome = config['data']['ome']
    k1 = config['data']['k1']
    k2 = config['data']['k2']
    k3 = config['data']['k3']
    wave_datamodel = DataModel(A, ome, k1, k2, k3)

    tmin, tmax = config['data']['tmin'], config['data']['tmax']
    xmin, xmax = config['data']['xmin'], config['data']['xmax']
    ymin, ymax = config['data']['ymin'], config['data']['ymax']
    zmin, zmax = config['data']['zmin'], config['data']['zmax']
    relative_noise = config['boundary']['relative_noise']

    wave_datagen = DataGenerator(
        wave_datamodel, 
        tmin, tmax, 
        xmin, xmax, 
        ymin, ymax, 
        zmin, zmax, 
        relative_noise=relative_noise
    )

    # boundary conditions
    N_bound = config['boundary']['N_bound']
    X0, u0, ut0 = wave_datagen.gen_bound(N_bound)

    # plotter.plot_bound(X0, u0, ut0)
    plotter.plot_bound_surface(X0, u0, N=N_bound)


    # training data
    N_train = config['train']['N_train']
    X_train = wave_datagen.gen_train(N_train)
    plotter.plot_sample(X0, u0, ut0, X_train)



    # --------------------------------------------------
    # create model
    # --------------------------------------------------

    # set hidden layers
    layers = config['model']['layers']

    # wave speed set from data model
    a1, a2, a3 = wave_datamodel.get_a()

    # data value boundary to init scalar layer
    lb, ub = wave_datagen.get_lb_ub()

    # create model
    model = WaveModel(layers, lb, ub, a1, a2, a3)

    # set model optimizer
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, 
        decay_steps=N_train // 10,
        decay_rate=0.94
    )
    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([600, 1000], [1e-3, 5e-4, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    # --------------------------------------------------
    # train model
    # --------------------------------------------------

    # set loss weights
    w_u0 = config['train']['w_u0']
    w_ut0 = config['train']['w_ut0']

    # set interation
    N_iter = config['train']['N_iter']

    # train
    model.fit(X0, u0, ut0, X_train, w_u0, w_ut0, N_iter=N_iter, optimizer=optimizer)

    hist_train = model.loss_train.copy()
    hist_val = model.loss_val.copy()
    hist_w = np.array(model.loss_weight.copy())

    print('saving loss history ...', flush=True)
    plotter.plot_loss_history(hist_train, show=False, title='LossTrain')
    plotter.plot_loss_history(hist_val, show=False, title='LossValidation')
    plotter.plot_loss_history(hist_w[:, 0], show=False, title='LossWeight1')
    plotter.plot_loss_history(hist_w[:, 1], show=False, title='LossWeight2')
    plotter.plot_loss_history(hist_w[:, 2], show=False, title='LossWeight3')

    # save model
    # model.save(os.path.join(save_dir, 'model.h5'))
    # print('model saved at {}'.format(os.path.join(save_dir, 'model.h5')), flush=True)
    

    # --------------------------------------------------
    # predict
    # --------------------------------------------------

    # set test data
    N_test = config['test']['N_test']
    N_time = int(config['test']['N_time'])

    ts = np.linspace(tmin, tmax, N_time)
    X_tests = []
    u_predicts = []
    u_trues = []

    for i, t_view in enumerate(ts):
        X_test = wave_datagen.gen_predict(N_test, t=t_view)

        # resize data
        u_predict = model.predict(X_test)
        u_true = wave_datamodel.u_model(X_test)

        X_tests.append(X_test)
        u_predicts.append(u_predict)
        u_trues.append(u_true)
    
    print('saving animated gif...', flush=True)
    plotter.plot_surface_animated(ts, X_tests, u_predicts, N_test, 'Predict3D')

    # get L2 loss at t=tmax
    u_predict = u_predicts[-1]
    u_true = u_trues[-1]
    L2_err = np.sum(np.square(u_predict - u_true)) / np.sum(np.square(u_true))

    print('All done!')
    return L2_err


def main():
    work_dir = './workspace1'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # set config
    config_base = {
        'name': 'plane_wave_3d',
        'randseed': 0,
        'data': {
            'model': 'plane_wave', 
            'A': 2.34,
            'ome': 3.45,
            'k1': 4.56,
            'k2': 4.56,
            'k3': 4.56,
            'tmin': 0.0,
            'tmax': 1.0,
            'xmin': -1.0,
            'xmax': 1.0,
            'ymin': -1.0,
            'ymax': 1.0,
            'zmin': -1.0,
            'zmax': 1.0,
        }, 
        'model': {
            'layers': [20, 20, 20, 20],
        },
        'boundary': {
            'N_bound': 50,
            'relative_noise': 1e-3,
        }, 
        'train': {
            'N_train': 1000,
            'N_iter': 2000,
            'w_u0': 1.0,
            'w_ut0': 1.0,
        },
        'test': {
            'N_test': 25,
            'N_time': 21,
        },
    }

    # set config
    all_layers = {
        '4x10': [10, 10, 10, 10],
        '4x20': [20, 20, 20, 20], 
        '4x40': [40, 40, 40, 40],
        # '6x10': [10, 10, 10, 10, 10, 10],
        # '6x20': [20, 20, 20, 20, 20, 20],
        # '6x40': [40, 40, 40, 40, 40, 40],
        # '8x10': [10, 10, 10, 10, 10, 10, 10, 10],
        # '8x20': [20, 20, 20, 20, 20, 20, 20, 20],
        # '8x40': [40, 40, 40, 40, 40, 40, 40, 40],
    }
    all_config = []
    for layers_name, layers in all_layers.items():
        config = copy.deepcopy(config_base)
        config['model']['layers'] = layers
        config['name'] = '{}_{}'.format(config['name'], layers_name)
        all_config.append(config)
    
    # train
    errs = []
    for config in all_config:
        print('-'*80)
        print('config: {}'.format(config['name']))
        print('-'*80)
        err = run_config(config, work_dir)
        errs.append(err)
        print('L2 error: {}'.format(err))
    
    # save errs
    with open(os.path.join(work_dir, 'errs.txt'), 'w') as f:
        for config, err in zip(all_config, errs):
            f.write('{}\t{}\n'.format(config['name'], err))



def run_config(config, work_dir):

    config_path = os.path.join(work_dir, 
        '{:s}.yaml'.format(config['name']))
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # TODO
    # apply circle wave formatted init boundary

    # run
    L2_err = run(config, work_dir)

    return L2_err


if __name__ == '__main__':
    main()
