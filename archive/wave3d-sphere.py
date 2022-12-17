'''
Author: Chenfa Zheng
Last Edit: 2022.04.22
'''


import tensorflow as tf
import numpy as np
from time import time
import os
import yaml
import imageio
import copy
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    
    def gen_bound_t(self, N_bound):
        # x0 = tf.random.uniform((N_bound, 1), minval=self.xmin, maxval=self.xmax, dtype=DTYPE)
        # y0 = tf.random.uniform((N_bound, 1), minval=self.ymin, maxval=self.ymax, dtype=DTYPE)
        # z0 = tf.random.uniform((N_bound, 1), minval=self.zmin, maxval=self.zmax, dtype=DTYPE)
        t0 = tf.random.uniform((N_bound, 1), minval=self.tmin, maxval=self.tmax, dtype=DTYPE)
        x0 = tf.zeros_like(t0)
        y0 = tf.zeros_like(t0)
        z0 = tf.zeros_like(t0)
        X0 = tf.concat([t0, x0, y0, z0], 1)
        
        return X0
    
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
    def __init__(self, A, ome, k):
        self.A = A
        self.ome = ome
        self.k = k
    
    def u_model(self, X):
        t = X[:, 0:1]
        x = X[:, 1:2]
        y = X[:, 2:3]
        z = X[:, 3:4]
        r = tf.sqrt(x**2 + y**2 + z**2)
        r = r + 1e-12
        
        u = self.A * tf.sin(self.ome * t - self.k * r) / r
        # print('shape test', flush=True)
        # print(u.shape, flush=True)
        # print(x.shape, flush=True)
        # print(y.shape, flush=True)
        # print(z.shape, flush=True)
        # print(t.shape, flush=True)
        # print(tf.sin(self.ome * t + self.k * r).shape, flush=True)
        # print((self.A * tf.sin(self.ome * t + self.k * r) / r).shape, flush=True)
        # sys.exit(0)
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
        a = self.ome / self.k
        return a


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
        umin = min([np.percentile(u.numpy(), 1) for u in us])
        umax = max([np.percentile(u.numpy(), 99) for u in us])
        # umin = min([tf.reduce_min(u) for u in us])
        # umax = max([tf.reduce_max(u) for u in us])
        umin = -1 if umin < -1 else umin
        umax = 1 if umax > 1 else umax
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

    
    def plot_uxyz(self, x, ux, y, uy, z, uz):
        fig = plt.figure(figsize=(9,6))
        ax1 = fig.add_subplot(311)
        ax1.plot(x, ux, '-', label='$u(x, y=0, z=0)$')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$u$')
        ax1.legend()
        ax2 = fig.add_subplot(312)
        ax2.plot(y, uy, '-', label='$u(x=0, y, z=0)$')
        ax2.set_xlabel('$y$')
        ax2.set_ylabel('$u$')
        ax2.legend()
        ax3 = fig.add_subplot(313)
        ax3.plot(z, uz, '-', label='$u(x=0, y=0, z)$')
        ax3.set_xlabel('$z$')
        ax3.set_ylabel('$u$')
        ax3.legend()
        plt.savefig('%s/uxyz.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        plt.close()

        # TODO make it animated!
    

    def plot_uxyz_animated(self, ts, x, uxs, y, uys, z, uzs, tag='Predict'):
        fig = plt.figure(figsize=(9,6))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        line1 = ax1.plot(x, uxs[0], '-', label='$u(x, y=0, z=0)$')[0]
        line2 = ax2.plot(y, uys[0], '-', label='$u(x=0, y, z=0)$')[0]
        line3 = ax3.plot(z, uzs[0], '-', label='$u(x=0, y=0, z)$')[0]

        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$u$')
        ax1.legend()
        ax2.set_xlabel('$y$')
        ax2.set_ylabel('$u$')
        ax2.legend()
        ax3.set_xlabel('$z$')
        ax3.set_ylabel('$u$')
        ax3.legend()

        fig_title = '$u(x, y, z)$ at t = %.2f'%ts[0]
        fig.suptitle(fig_title)

        def animate(i):
            line1.set_ydata(uxs[i])
            line2.set_ydata(uys[i])
            line3.set_ydata(uzs[i])

            fig_title = tag + '\t' + '$u(x, y, z)$ at t = %.2f'%ts[i]
            fig.suptitle(fig_title)

            return line1, line2, line3
        
        anim = animation.FuncAnimation(fig, animate, frames=len(ts), interval=100)
        writer = animation.writers['ffmpeg']
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('%s/uxyz_animated_%s.gif'%(self.save_dir, tag), writer=writer)

        plt.close()
    

    def plot_uxyz_animated_both(self, ts, x, uxs, uxs2, y, uys, uys2, z, uzs, uzs2):
        fig = plt.figure(figsize=(9,6))
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        line1 = ax1.plot(x, uxs[0], '-', label='$u(x, y=0, z=0)$')[0]
        line2 = ax2.plot(y, uys[0], '-', label='$u(x=0, y, z=0)$')[0]
        line3 = ax3.plot(z, uzs[0], '-', label='$u(x=0, y=0, z)$')[0]
        line11 = ax1.plot(x, uxs2[0], '--', label='$u(x, y=0, z=0)$')[0]
        line22 = ax2.plot(y, uys2[0], '--', label='$u(x=0, y, z=0)$')[0]
        line33 = ax3.plot(z, uzs2[0], '--', label='$u(x=0, y=0, z)$')[0]

        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$u$')
        ax1.legend()
        ax2.set_xlabel('$y$')
        ax2.set_ylabel('$u$')
        ax2.legend()
        ax3.set_xlabel('$z$')
        ax3.set_ylabel('$u$')
        ax3.legend()

        fig_title = '$u(x, y, z)$ at t = %.2f'%ts[0]
        fig.suptitle(fig_title)

        def animate(i):
            line1.set_ydata(uxs[i])
            line2.set_ydata(uys[i])
            line3.set_ydata(uzs[i])
            line11.set_ydata(uxs2[i])
            line22.set_ydata(uys2[i])
            line33.set_ydata(uzs2[i])

            fig_title = '$u(x, y, z)$ at t = %.2f'%ts[i]
            fig.suptitle(fig_title)

            return line1, line2, line3, line11, line22, line33
        
        anim = animation.FuncAnimation(fig, animate, frames=len(ts), interval=100)
        writer = animation.writers['ffmpeg']
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('%s/uxyz_animated_%s.gif'%(self.save_dir, 'Both'), writer=writer)

        plt.close()

    
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
    def __init__(self, layers, lb, ub, a):
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
        self.a = a
    
    def predict(self, X):
        return self.call(X)

    # residual of wave equation
    def _func_r(self, u_tt, u_xx, u_yy, u_zz):
        return u_tt - self.a * (u_xx + u_yy + u_zz)
    
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
    
    def compute_loss(self, X0, u0, ut0, Xt0, X_train, w_u0, w_ut0, w_ui0, X0_val, u0_val, ut0_val, Xt0_val):
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

        # boundary loss at u_i(t, i=0)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(Xt0)
            tape.watch(Xt0_val)
            u0t0_pred = self.call(Xt0)
            u0t0_pred_val = self.call(Xt0_val)
        uxt0_pred = tape.batch_jacobian(u0t0_pred, Xt0[:, 1:2])[:, :, 0]
        uyt0_pred = tape.batch_jacobian(u0t0_pred, Xt0[:, 2:3])[:, :, 0]
        uzt0_pred = tape.batch_jacobian(u0t0_pred, Xt0[:, 3:4])[:, :, 0]
        uxt0_pred_val = tape.batch_jacobian(u0t0_pred_val, Xt0_val[:, 1:2])[:, :, 0]
        uyt0_pred_val = tape.batch_jacobian(u0t0_pred_val, Xt0_val[:, 2:3])[:, :, 0]
        uzt0_pred_val = tape.batch_jacobian(u0t0_pred_val, Xt0_val[:, 3:4])[:, :, 0]
        l3 = tf.reduce_mean(tf.square(uxt0_pred) + tf.square(uyt0_pred) + tf.square(uzt0_pred))
        l3_val = tf.reduce_mean(tf.square(uxt0_pred_val) + tf.square(uyt0_pred_val) + tf.square(uzt0_pred_val))

        # loss summary
        loss = l0 + w_u0 * l1 + w_ut0 * l2 + w_ui0 * l3
        loss_val = w_u0 * l1_val + w_ut0 * l2_val + w_ui0 * l3_val
        return loss, loss_val
    
    def get_grad(self, X0, u0, ut0, Xt0, X_train, w_u0, w_ut0, w_ui0, X0_val, u0_val, ut0_val, Xt0_val):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            loss, loss_val = self.compute_loss(X0, u0, ut0, Xt0, X_train, w_u0, w_ut0, w_ui0, X0_val, u0_val, ut0_val, Xt0_val)
        grad = tape.gradient(loss, self.trainable_variables)
        return loss, grad, loss_val
    
    def fit(self, X0, u0, ut0, Xt0, X_train, w_u0, w_ut0, w_ui0, N_iter=1000, optimizer=None):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # split data: 80% for train, 20% for validation
        X0_train, X0_val = tf.split(X0, [int(0.8*X0.shape[0]), int(0.2*X0.shape[0])], axis=0)
        u0_train, u0_val = tf.split(u0, [int(0.8*u0.shape[0]), int(0.2*u0.shape[0])], axis=0)
        ut0_train, ut0_val = tf.split(ut0, [int(0.8*ut0.shape[0]), int(0.2*ut0.shape[0])], axis=0)
        Xt0_train, Xt0_val = tf.split(Xt0, [int(0.8*Xt0.shape[0]), int(0.2*Xt0.shape[0])], axis=0)

        @tf.function
        def train_step():
            loss, grad_theta, loss_val = self.get_grad(X0_train, u0_train, ut0_train, Xt0_train, X_train, w_u0, w_ut0, w_ui0, X0_val, u0_val, ut0_val, Xt0_val)
            optimizer.apply_gradients(zip(grad_theta, self.trainable_variables))
            return loss, loss_val

        hist_train = []
        hist_val = []
        t0 = time()

        for i in range(N_iter+1):
            loss_train, loss_val = train_step()
            # loss_b1, loss_b2 = self.get_bound_r(X0_val, u0_val, ut0_val)
            hist_train.append(loss_train.numpy())
            # hist_val.append(w_u0 * loss_b1.numpy() + w_ut0 * loss_b2.numpy())
            hist_val.append(loss_val.numpy())

            if i%10 == 0:
                print('iteration: {:05d}, train loss: {:10.8e} test bound loss: {:10.8e}'.format(i, loss_train, loss_val))

        print('time: {:.2f} sec'.format(time() - t0))
        print('final train loss: {:10.8e}'.format(loss_train))

        return hist_train, hist_val


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
    k = config['data']['k']
    wave_datamodel = DataModel(A, ome, k)

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
    Xt0 = wave_datagen.gen_bound_t(N_bound)

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
    a = wave_datamodel.get_a()

    # data value boundary to init scalar layer
    lb, ub = wave_datagen.get_lb_ub()

    # create model
    model = WaveModel(layers, lb, ub, a)

    # set model optimizer
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2, 
        decay_steps=N_train,
        decay_rate=0.96
    )
    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([600, 1000], [1e-3, 5e-4, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    # --------------------------------------------------
    # train model
    # --------------------------------------------------

    # set loss weights
    w_u0 = config['train']['w_u0']
    w_ut0 = config['train']['w_ut0']
    w_ui0 = config['train']['w_ui0']

    # set interation
    N_iter = config['train']['N_iter']

    # train
    hist_train, hist_val = model.fit(X0, u0, ut0, Xt0, X_train, w_u0, w_ut0, w_ui0, N_iter=N_iter, optimizer=optimizer)

    print('saving loss history ...', flush=True)
    plotter.plot_loss_history(hist_train, show=False, title='LossTrain')
    plotter.plot_loss_history(hist_val, show=False, title='LossValidation')

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
    x_test = tf.reshape(tf.linspace(xmin, xmax, N_test), [N_test, 1])
    y_test = tf.reshape(tf.linspace(ymin, ymax, N_test), [N_test, 1])
    z_test = tf.reshape(tf.linspace(zmin, zmax, N_test), [N_test, 1])
    zero_test = tf.zeros([N_test, 1])
    uxs, uys, uzs = [], [], []
    uxs_true, uys_true, uzs_true = [], [], []

    for i, t_view in enumerate(ts):
        X_test = wave_datagen.gen_predict(N_test, t=t_view)

        # resize data
        u_predict = model.predict(X_test)
        u_true = wave_datamodel.u_model(X_test)

        X_tests.append(X_test)
        u_predicts.append(u_predict)
        u_trues.append(u_true)

        # another test
        t_test = tf.ones((N_test, 1)) * t_view
        Xx_test = tf.concat([t_test, x_test, zero_test, zero_test], axis=1)
        ux_predict = model.predict(Xx_test)
        ux_true = wave_datamodel.u_model(Xx_test)
        Xy_test = tf.concat([t_test, zero_test, y_test, zero_test], axis=1)
        uy_predict = model.predict(Xy_test)
        uy_true = wave_datamodel.u_model(Xy_test)
        Xz_test = tf.concat([t_test, zero_test, zero_test, z_test], axis=1)
        uz_predict = model.predict(Xz_test)
        uz_true = wave_datamodel.u_model(Xz_test)

        uxs.append(ux_predict.numpy())
        uys.append(uy_predict.numpy())
        uzs.append(uz_predict.numpy())
        uxs_true.append(ux_true.numpy())
        uys_true.append(uy_true.numpy())
        uzs_true.append(uz_true.numpy())
    
    print('saving animated gif...', flush=True)
    plotter.plot_surface_animated(ts, X_tests, u_predicts, N_test, 'Predict3D')
    plotter.plot_surface_animated(ts, X_tests, u_trues, N_test, 'True3D')
    plotter.plot_uxyz_animated(ts, x_test.numpy(), uxs, y_test.numpy(), uys, z_test.numpy(), uzs, tag='Predict')
    plotter.plot_uxyz_animated(ts, x_test.numpy(), uxs_true, y_test.numpy(), uys_true, z_test.numpy(), uzs_true, tag='True')
    plotter.plot_uxyz_animated_both(ts, x_test.numpy(), uxs, uxs_true, y_test.numpy(), uys, uys_true, z_test.numpy(), uzs, uzs_true)

    # get L2 loss at t=tmax
    u_predict = u_predicts[-1]
    u_true = u_trues[-1]
    L2_norm = tf.norm(u_predict - u_true, ord=2, axis=1)
    L2_norm_mean = tf.reduce_mean(L2_norm)
    L2_norm_std = tf.math.reduce_std(L2_norm)

    # np.save(os.path.join(save_dir, 'history_train.npy'), hist_train)
    # np.save(os.path.join(save_dir, 'history_val.npy'), hist_val)
    np.savetxt(os.path.join(save_dir, 'history_train.txt'), hist_train)
    np.savetxt(os.path.join(save_dir, 'history_val.txt'), hist_val)

    print('All done!')
    return L2_norm_mean, L2_norm_std


def main():
    work_dir = './workspace5'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # set config
    config_base = {
        'name': 'sphere_wave_3d',
        'randseed': 0,
        'data': {
            'model': 'sphere_wave', 
            'A': 1.23,
            'ome': 4.56,
            'k': 7.89,
            'tmin': 0.0,
            'tmax': 2.0,
            'xmin': -2.0,
            'xmax': 2.0,
            'ymin': -2.0,
            'ymax': 2.0,
            'zmin': -2.0,
            'zmax': 2.0,
        }, 
        'model': {
            'layers': [20, 20, 20, 20],
        },
        'boundary': {
            'N_bound': 100,
            'relative_noise': 1e-3,
        }, 
        'train': {
            'N_train': 1000,
            'N_iter': 50,
            'w_u0': 1.0,
            'w_ut0': 1.0,
            'w_ui0': 1.0,
        },
        'test': {
            'N_test': 25,
            'N_time': 3,
        },
    }

    # set config
    all_layers = {
        '4x10': [10, 10, 10, 10],
        '4x20': [20, 20, 20, 20], 
        '4x40': [40, 40, 40, 40],
        '6x10': [10, 10, 10, 10, 10, 10],
        '6x20': [20, 20, 20, 20, 20, 20],
        '6x40': [40, 40, 40, 40, 40, 40],
    }
    all_config = []
    for layers_name, layers in all_layers.items():
        config = copy.deepcopy(config_base)
        config['model']['layers'] = layers
        config['name'] = '{}_{}'.format(config['name'], layers_name)
        all_config.append(config)
    
    # train
    errs = []
    err_stds = []
    for config in all_config:
        print('-'*80)
        print('config: {}'.format(config['name']))
        print('-'*80)
        err_mean, err_std = run_config(config, work_dir)
        errs.append(err_mean)
        err_stds.append(err_std)
        print('L2 error: {} std: {}'.format(err_mean, err_std))
    
    # save errs
    with open(os.path.join(work_dir, 'errs.txt'), 'w') as f:
        for config, err, err_std in zip(all_config, errs, err_stds):
            f.write('{}\t{}\t{}\n'.format(config['name'], err, err_std))



def run_config(config, work_dir):

    config_path = os.path.join(work_dir, 
        '{:s}.yaml'.format(config['name']))
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # TODO
    # apply circle wave formatted init boundary

    # run
    L2_mean, L2_std = run(config, work_dir)

    return L2_mean, L2_std


if __name__ == '__main__':
    main()
