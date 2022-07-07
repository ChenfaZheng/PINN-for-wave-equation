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
    def __init__(self, model, tmin, tmax, xmin, xmax, ymin, ymax, relative_noise=0.1):
        self.model = model
        self.tmin = tmin
        self.tmax = tmax
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.nr = relative_noise
    
    def _gen_relative_noise(self, n):
        return tf.random.normal((n, 1), dtype=DTYPE) * self.nr
    
    def get_lb_ub(self):
        lb = tf.constant([self.tmin, self.xmin, self.ymin], dtype=DTYPE)
        ub = tf.constant([self.tmax, self.xmax, self.ymax], dtype=DTYPE)
        return lb, ub
    
    def gen_bound(self, N_bound):
        # bound where t == 0
        # x0 = tf.random.uniform((N_bound, 1), minval=self.xmin, maxval=self.xmax, dtype=DTYPE)
        x0 = tf.reshape(tf.linspace(self.xmin, self.xmax, N_bound), (N_bound, 1))
        y0 = tf.reshape(tf.linspace(self.ymin, self.ymax, N_bound), (N_bound, 1))

        xx0, yy0 = tf.meshgrid(x0, y0)
        xx0 = tf.reshape(xx0, (N_bound**2, 1))
        yy0 = tf.reshape(yy0, (N_bound**2, 1))
        tt0 = tf.zeros_like(xx0) + self.tmin

        X0 = tf.concat([tt0, xx0, yy0], 1)

        u0 = self.model.u0(X0) * (1.0 + self._gen_relative_noise(N_bound**2))
        ut0 = self.model.ut0(X0) * (1.0 + self._gen_relative_noise(N_bound**2))
        
        return X0, u0, ut0
    
    def gen_bound_t(self, N_bound):
        # bound at border
        t0 = tf.random.uniform((N_bound, 1), minval=self.tmin, maxval=self.tmax, dtype=DTYPE)
        x0 = tf.zeros_like(t0)
        y0 = tf.zeros_like(t0)
        X0 = tf.concat([t0, x0, y0], 1)
        return X0
    
    def gen_train(self, N_train):
        t_train = tf.random.uniform((N_train, 1), minval=self.tmin, maxval=self.tmax, dtype=DTYPE)
        x_train = tf.random.uniform((N_train, 1), minval=self.xmin, maxval=self.xmax, dtype=DTYPE)
        y_train = tf.random.uniform((N_train, 1), minval=self.ymin, maxval=self.ymax, dtype=DTYPE)
        X_train = tf.concat([t_train, x_train, y_train], 1)
        return X_train
    
    def gen_predict(self, N_predict, t=None):
        x = tf.reshape(tf.cast(tf.linspace(self.xmin, self.xmax, N_predict), dtype=DTYPE), (N_predict, 1))
        y = tf.reshape(tf.cast(tf.linspace(self.ymin, self.ymax, N_predict), dtype=DTYPE), (N_predict, 1))
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (N_predict**2, 1))
        yy = tf.reshape(yy, (N_predict**2, 1))
        if t is None:
            t_view = self.tmax
        else:
            t_view = t
        tt = tf.ones_like(xx) * t_view
        # tt = tf.random.uniform((N_predict**3, 1), minval=self.tmin, maxval=self.tmax, dtype=DTYPE)
        X = tf.concat([tt, xx, yy], 1)
        return X


class DataModel():
    def __init__(self, a):
        self.a = a
    
    def u_forward(self, X):
        t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        u = tf.exp(-(x - self.a*t)**2 - (y - self.a*t)**2)
        return u * 0.5
    
    def u_reflection2(self, X):
        t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        u = tf.exp(-(x + self.a*t)**2 - (y + self.a*t)**2)
        return u * 0.5

    def u_model(self, X):
        return self.u_forward(X) + self.u_reflection2(X)
    
    def u0(self, X):
        # t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        # u0 = tf.exp(-tf.pow(x, 2))
        # u0 = self.u_model(X)
        t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        u0 = tf.exp(-(x**2 + y**2))
        return u0
    
    def ut0(self, X):
        # x = X[:, 1:2]
        # ut0 = 2*self.a*x * self.u_model(X)
        # # return ut0
        # return - self.u_model(X)
        return tf.zeros_like(self.u0(X))
    
    def get_a(self):
        return self.a


class Plotter():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def plot_bound(self, X0, u0, ut0):
        t0, x0 = X0[:, 0:1], X0[:, 1:2]

        fig = plt.figure(figsize=(9,6))
        fig.suptitle('Initial Bound')
        ax1 = fig.add_subplot(211)
        ax1.scatter(x0, u0, alpha=0.5, marker='.')
        ax1.set_xlabel('$x$')
        ax1.set_ylabel('$u$')
        ax2 = fig.add_subplot(212)
        ax2.scatter(x0, ut0, alpha=0.5, marker='.')
        ax2.set_xlabel('$x$')
        ax2.set_ylabel('$u_t$')
        plt.savefig('%s/InitBound.jpg'%self.save_dir, bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_contour(self, X, u, N, title='Contour'):
        t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
        u = tf.reshape(u, (N, N))
        t = tf.reshape(t, (N, N))
        x = tf.reshape(x, (N, N))
        y = tf.reshape(y, (N, N))

        fig = plt.figure(figsize=(9,6))
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        cax = ax.contourf(x, y, u, 100)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.colorbar(cax, ax=ax, fraction=0.1, pad=0.1)
        plt.savefig('%s/%s.jpg'%(self.save_dir, title), bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_animated_line(self, ts, X_tests, u_predicts, u_trues, N_test, title='Predict'):
        umax_p = max([tf.reduce_max(u_pred) for u_pred in u_predicts])
        umin_p = min([tf.reduce_min(u_pred) for u_pred in u_predicts])
        umax_t = max([tf.reduce_max(u_true) for u_true in u_trues])
        umin_t = min([tf.reduce_min(u_true) for u_true in u_trues])
        umax, umin = max(umax_p, umax_t), min(umin_p, umin_t)

        fig = plt.figure(figsize=(9,6))
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u$')
        ax.set_ylim([umin, umax])

        line1 = ax.plot(X_tests[0][:, 1:2], u_predicts[0], 'r-', lw=2, label='predict')[0]
        line2 = ax.plot(X_tests[0][:, 1:2], u_trues[0], 'b--', lw=2, alpha=0.6, label='true')[0]

        ax.legend(loc='upper left')

        def animate(i):
            x = X_tests[i][:, 1:2]
            t = ts[i]
            line1.set_data(x, u_predicts[i])
            line2.set_data(x, u_trues[i])

            fig.suptitle('t = %.2f'%t)
        
        anim = animation.FuncAnimation(fig, animate, frames=len(ts), interval=100)
        writer = animation.writers['ffmpeg']
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('%s/%s_animated.gif'%(self.save_dir, title), writer=writer)
        plt.close()
    
    def plot_animated_contour(self, ts, X_tests, u_predicts, N_test, title='Predict'):
        umax, umin = max([tf.reduce_max(u_pred) for u_pred in u_predicts]), min([tf.reduce_min(u_pred) for u_pred in u_predicts])

        fig = plt.figure(figsize=(9,6))
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')

        x0 = X_tests[0][:, 1:2]
        y0 = X_tests[0][:, 2:3]
        u0 = u_predicts[0]

        xx = tf.reshape(x0, (N_test, N_test))
        yy = tf.reshape(y0, (N_test, N_test))
        uu = tf.reshape(u0, (N_test, N_test))

        cax = ax.contourf(xx, yy, uu, 100, vmin=umin, vmax=umax)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        fig.colorbar(cax, ax=ax, fraction=0.1, pad=0.1)

        def animate(i):
            t = ts[i]
            ax.clear()
            x0 = X_tests[i][:, 1:2]
            y0 = X_tests[i][:, 2:3]
            u0 = u_predicts[i]
            xx = tf.reshape(x0, (N_test, N_test))
            yy = tf.reshape(y0, (N_test, N_test))
            uu = tf.reshape(u0, (N_test, N_test))
            ax.contourf(xx, yy, uu, 100, vmin=umin, vmax=umax)
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')

            fig.suptitle('t = %.2f'%t)
        
        anim = animation.FuncAnimation(fig, animate, frames=len(ts), interval=100)
        writer = animation.writers['ffmpeg']
        writer = writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('%s/%s_animated.gif'%(self.save_dir, title), writer=writer)
    
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
        # 3 dimensional input: t, x, y
        self.add(tf.keras.Input(3))
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
    def _func_r(self, u_tt, u_xx, u_yy):
        return u_tt - self.a * (u_xx + u_yy)
    
    # get residual of wave equation
    def get_r(self, X):
        with tf.GradientTape(persistent=True) as tape:
            t, x, y = X[:, 0:1], X[:, 1:2], X[:, 2:3]
            tape.watch(t)
            tape.watch(x)
            tape.watch(y)

            u = self.call(tf.stack([t[:, 0], x[:, 0], y[:, 0]], axis=1))
            u_t = tape.gradient(u, t)
            u_x = tape.gradient(u, x)
            u_y = tape.gradient(u, y)

        u_tt = tape.gradient(u_t, t)
        u_xx = tape.gradient(u_x, x)
        u_yy = tape.gradient(u_y, y)

        r = self._func_r(u_tt, u_xx, u_yy)
        return r
    
    def compute_loss(self, X0, u0, ut0, Xb, X_train, w_u0, w_ut0, w_ub, X0_val, u0_val, ut0_val, Xb_val):
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

        # boundary loss at border like u_x(t, x=xmax)
        # loss set as u_x(t, x=0, y=0) == 0 and u_y(t, x=0, y=0) == 0
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(Xb)
            tape.watch(Xb_val)
            ub_pred = self.call(Xb)
            ub_pred_val = self.call(Xb_val)
        ubx_pred = tape.batch_jacobian(ub_pred, Xb)[:, :, 1]
        ubx_pred_val = tape.batch_jacobian(ub_pred_val, Xb_val)[:, :, 1]
        uby_pred = tape.batch_jacobian(ub_pred, Xb)[:, :, 2]
        uby_pred_val = tape.batch_jacobian(ub_pred_val, Xb_val)[:, :, 2]
        l3 = tf.reduce_mean(tf.square(ubx_pred) + tf.square(uby_pred))
        l3_val = tf.reduce_mean(tf.square(ubx_pred_val) + tf.square(uby_pred_val))

        # loss summary
        loss = l0 + w_u0 * l1 + w_ut0 * l2 + w_ub * l3
        loss_val = w_u0 * l1_val + w_ut0 * l2_val + w_ub * l3_val
        return loss, loss_val
    
    def get_grad(self, X0, u0, ut0, Xb, X_train, w_u0, w_ut0, w_ub, X0_val, u0_val, ut0_val, Xb_val):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            loss, loss_val = self.compute_loss(X0, u0, ut0, Xb, X_train, w_u0, w_ut0, w_ub, X0_val, u0_val, ut0_val, Xb_val)
        grad = tape.gradient(loss, self.trainable_variables)
        return grad, loss, loss_val
    
    def fit(self, X0, u0, ut0, Xb, X_train, w_u0, w_ut0, w_ub, N_iter=1000, optimizer=None):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        
        # split data: 80% for train, 20% for validation
        X0_train, X0_val = tf.split(X0, [int(0.8*X0.shape[0]), int(0.2*X0.shape[0])], axis=0)
        u0_train, u0_val = tf.split(u0, [int(0.8*u0.shape[0]), int(0.2*u0.shape[0])], axis=0)
        ut0_train, ut0_val = tf.split(ut0, [int(0.8*ut0.shape[0]), int(0.2*ut0.shape[0])], axis=0)
        Xb_train, Xb_val = tf.split(Xb, [int(0.8*Xb.shape[0]), int(0.2*Xb.shape[0])], axis=0)

        @tf.function
        def train_step():
            grad_theta, loss, loss_val = self.get_grad(X0_train, u0_train, ut0_train, Xb_train, X_train, w_u0, w_ut0, w_ub, X0_val, u0_val, ut0_val, Xb_val)
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

    a = 1
    wave_datamodel = DataModel(a)

    tmin, tmax = config['data']['tmin'], config['data']['tmax']
    xmin, xmax = config['data']['xmin'], config['data']['xmax']
    ymin, ymax = config['data']['ymin'], config['data']['ymax']
    relative_noise = config['boundary']['relative_noise']

    wave_datagen = DataGenerator(
        wave_datamodel, 
        tmin, tmax, 
        xmin, xmax, 
        ymin, ymax, 
        relative_noise=relative_noise
    )

    # boundary conditions
    N_bound = config['boundary']['N_bound']
    X0, u0, ut0 = wave_datagen.gen_bound(N_bound)
    Xb = wave_datagen.gen_bound_t(N_bound)

    # plotter.plot_bound(X0, u0, ut0)
    plotter.plot_contour(X0, u0, N_bound, 'Boundary_u0')
    plotter.plot_contour(X0, ut0, N_bound, 'Boundary_ut0')

    # training data
    N_train = config['train']['N_train']
    X_train = wave_datagen.gen_train(N_train)
    # plotter.plot_sample(X0, u0, ut0, Xb, X_train)



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
        decay_steps=1000,
        decay_rate=0.60
    )
    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([600, 1000], [1e-3, 5e-4, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)


    # --------------------------------------------------
    # train model
    # --------------------------------------------------

    # set loss weights
    w_u0 = config['train']['w_u0']
    w_ut0 = config['train']['w_ut0']
    w_ub = config['train']['w_ub']

    # set interation
    N_iter = config['train']['N_iter']

    # train
    hist_train, hist_val = model.fit(X0, u0, ut0, Xb, X_train, w_u0, w_ut0, w_ub, N_iter=N_iter, optimizer=optimizer)

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

    for i, t_view in enumerate(ts):
        X_test = wave_datagen.gen_predict(N_test, t=t_view)

        # resize data
        u_predict = model.predict(X_test)
        u_true = wave_datamodel.u_model(X_test)

        X_tests.append(X_test)
        u_predicts.append(u_predict)
        u_trues.append(u_true)
    
    print('saving animated gif...', flush=True)
    # plotter.plot_animated_line(ts, X_tests, u_predicts, u_trues, N_test, 'Predict')
    plotter.plot_animated_contour(ts, X_tests, u_predicts, N_test, 'Predict')
    # plotter.plot_animated_contour(ts, X_tests, u_trues, N_test, 'True')
    # plotter.plot_animated_contour(ts, X_tests, [u_p - u_t for u_p, u_t in zip(u_predicts, u_trues)], N_test, 'Residuals')

    # L2 loss
    L2_norms = [tf.norm(u_predict - u_true, ord=2) for u_predict, u_true in zip(u_predicts, u_trues)]
    L2_mean = tf.reduce_mean(L2_norms)
    L2_std = tf.math.reduce_std(L2_norms)

    # np.save(os.path.join(save_dir, 'history_train.npy'), hist_train)
    # np.save(os.path.join(save_dir, 'history_val.npy'), hist_val)
    np.savetxt(os.path.join(save_dir, 'history_train.txt'), hist_train)
    np.savetxt(os.path.join(save_dir, 'history_val.txt'), hist_val)

    print('All done!')

    return L2_mean, L2_std, hist_train[-1], hist_val[-1]


def main():
    work_dir = './gaussian-2d-new'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # set config
    config_base = {
        'name': 'reflection2',
        'randseed': 0,
        'data': {
            'model': 'gaussian-2d', 
            'a': 1.0,
            'tmin': 0.0,
            'tmax': 10.0,
            'xmin': -5.0,
            'xmax': 5.0,
            'ymin': -5.0,
            'ymax': 5.0,
        }, 
        'model': {
            'layers': [20, 20, 20, 20],
        },
        'boundary': {
            'N_bound': 1000,
            'relative_noise': 0.0,
        }, 
        'train': {
            'N_train': 10000,
            'N_iter': 10000,
            'w_u0': 1.0,
            'w_ut0': 1.0,
            'w_ub': 1.0,
        },
        'test': {
            'N_test': 1000,
            'N_time': 21,
        },
    }

    # set config
    all_layers = {
        '3x10': [10, 10, 10],
        '3x20': [20, 20, 20],
        '3x30': [30, 30, 30],
        '4x10': [10, 10, 10, 10],
        '4x20': [20, 20, 20, 20], 
        '4x30': [30, 30, 30, 30],
        '5x10': [10, 10, 10, 10, 10],
        '5x20': [20, 20, 20, 20, 20],
        '5x30': [30, 30, 30, 30, 30],
    }
    all_config = []
    for layers_name, layers in all_layers.items():
        config = copy.deepcopy(config_base)
        config['model']['layers'] = layers
        config['name'] = '{}_{}_{}_{}'.format(config['name'], layers_name, config['train']['N_iter'], config['boundary']['relative_noise'])
        all_config.append(config)
    
    # train
    errs = []
    errstds = []
    loss_trains = []
    loss_vals = []
    for config in all_config:
        print('-'*80)
        print('config: {}'.format(config['name']))
        print('-'*80)
        err, errstd, loss_train, loss_val = run_config(config, work_dir)
        
        errs.append(err)
        errstds.append(errstd)
        loss_trains.append(loss_train)
        loss_vals.append(loss_val)
    
    # save
    with open(os.path.join(work_dir, 'errs-gaussian-2d-reflection2-{}-{}.csv'.format(config['train']['N_iter'], config['boundary']['relative_noise'])), 'w') as f:
        f.write('name,err_mean,err_std,loss_train,loss_validation\n')
        for config, err, errstd, lt, lv in zip(all_config, errs, errstds, loss_trains, loss_vals):
            f.write('{},{},{},{},{}\n'.format(config['name'], err, errstd, lt, lv))



def run_config(config, work_dir):

    config_path = os.path.join(work_dir, 
        '{:s}-{}-{}.yaml'.format(config['name'], config['train']['N_iter'], config['boundary']['relative_noise']))
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # run
    err, errstd, lt, lv = run(config, work_dir)

    return err, errstd, lt, lv


if __name__ == '__main__':
    main()
