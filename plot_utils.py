import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import torch
from torch_util import DEVICE
import matplotlib.patheffects as pe
import pandas as pd

def updatePLT(W, l=4, w=3, fontsize=10):
    plt.rcParams.update({
        'figure.figsize': (W, W/(l/w)),     # 4:3 aspect ratio
        'font.size' : fontsize,                   # Set font size to 11pt
        'axes.labelsize': fontsize,               # -> axis labels
        'legend.fontsize': fontsize,              # -> legends
        'font.family': 'sans-serif',
        'text.usetex': True,
        'text.latex.preamble': (            # LaTeX preamble
            r'\usepackage{lmodern}'
            # ... more packages if needed
        )
    })

def compare_control_results_jumps_paper(ts, 
                                        x_nolearnings, us_nolearnings,
                                        x_priors, us_priors,
                                        x_mpcs, us_mpcs, 
                                        names, jumps, fn=None):    
    fig, axes = plt.subplots(1, 1, sharex=True, constrained_layout=True)
    
    axes.yaxis.tick_right()
    axes.yaxis.set_label_position("right")
    axes.set_ylabel('control')
    
    axes.plot(ts, us_priors.mean(axis=0), lw=.8, color='tab:orange')
    axes.plot(ts, us_nolearnings.mean(axis=0), lw=.8, color='tab:red')
    axes.plot(ts, us_mpcs.mean(axis=0), lw=.8, color='tab:blue')
    
    if jumps is not None:
        for i in range(len(jumps)):
            if jumps[i]>0:
                axes.axvline(ts[i], ls='--', lw=1.5, color='gray', alpha=0.8)

    if jumps is None:
        lgd = fig.legend(names, framealpha=0.4, handlelength=0.6, 
                   loc='upper center', bbox_to_anchor=(0.45, 0.06), 
                   fancybox=True, ncol=4)
    else:
        lgd = fig.legend(names + ['jumps'], framealpha=0.4, handlelength=0.6, 
                   loc='upper center', bbox_to_anchor=(0.45, 0.06), 
                   fancybox=True, ncol=4)
        
    for legobj in lgd.legendHandles:
        legobj.set_linewidth(2.0)
    
    
    if True:
        axes.fill_between(ts, us_priors.mean(axis=0)-us_priors.std(axis=0), 
                                 us_priors.mean(axis=0)+us_priors.std(axis=0), 
                                     lw=1., color='tab:orange', alpha=0.3)
        axes.fill_between(ts, us_nolearnings.mean(axis=0)-us_nolearnings.std(axis=0), 
                                 us_nolearnings.mean(axis=0)+us_nolearnings.std(axis=0), 
                                     lw=1., color='tab:red', alpha=0.3)
        axes.fill_between(ts, us_mpcs.mean(axis=0)-us_mpcs.std(axis=0), 
                                 us_mpcs.mean(axis=0)+us_mpcs.std(axis=0), 
                                     lw=1., color='tab:blue', alpha=0.3)
    else:
        axes.fill_between(ts, us_priors.min(axis=0), us_priors.max(axis=0), lw=1., color='tab:orange', alpha=0.3)
        axes.fill_between(ts, us_nolearnings.min(axis=0), us_nolearnings.max(axis=0), lw=1., color='tab:red', alpha=0.3)
        axes.fill_between(ts, us_mpcs.min(axis=0),  us_mpcs.max(axis=0), lw=1., color='tab:blue', alpha=0.3)
        
        
    
    plt.tight_layout()
    if fn is not None: plt.savefig(fn, bbox_inches='tight')
    plt.show()
    return axes


def compare_control_results_jumps_paper2(ts, 
                                        x_nolearnings, us_nolearnings,
                                        x_priors, us_priors,
                                        x_mpcs, us_mpcs, 
                                        x_mpc_withs, us_mpc_withs,  
                                        names, jumps, fn=None):    
    fig, axes = plt.subplots(1, 1, sharex=True, constrained_layout=True)
    axes.yaxis.tick_right()
    axes.yaxis.set_label_position("right")
    axes.set_ylabel('control')
    
    if jumps is not None:
        for i in range(len(jumps)):
            if jumps[i]>0:
                axes.axvline(ts[i], ls='--', lw=1.5, color='gray', alpha=0.8)
    
    axes.plot(ts, us_priors.mean(axis=0), lw=.8, color='tab:orange')
    axes.plot(ts, us_mpcs.mean(axis=0), lw=.8, color='tab:blue')
    axes.plot(ts, us_mpc_withs.mean(axis=0), lw=.8, color='tab:green')
    axes.plot(ts, us_nolearnings.mean(axis=0), lw=.8, color='tab:red')

    axes.fill_between(ts, us_priors.mean(axis=0)-us_priors.std(axis=0), 
                             us_priors.mean(axis=0)+us_priors.std(axis=0), 
                                 lw=1., color='tab:orange', alpha=0.3)
    axes.fill_between(ts, us_mpcs.mean(axis=0)-us_mpcs.std(axis=0), 
                             us_mpcs.mean(axis=0)+us_mpcs.std(axis=0), 
                                 lw=1., color='tab:blue', alpha=0.3)
    axes.fill_between(ts, us_mpc_withs.mean(axis=0)-us_mpc_withs.std(axis=0), 
                             us_mpc_withs.mean(axis=0)+us_mpc_withs.std(axis=0), 
                                 lw=1., color='tab:green', alpha=0.3)
    axes.fill_between(ts, us_nolearnings.mean(axis=0)-us_nolearnings.std(axis=0), 
                             us_nolearnings.mean(axis=0)+us_nolearnings.std(axis=0), 
                                 lw=1., color='tab:red', alpha=0.3)
    if jumps is None:
        lgd = fig.legend(names, framealpha=0.4, handlelength=0.6, 
                   loc='upper center', bbox_to_anchor=(0.5, 0.06), 
                   fancybox=True, ncol=4)
    else:
        lgd = fig.legend(names + ['jumps'], framealpha=0.4, handlelength=0.6, 
                   loc='upper center', bbox_to_anchor=(0.45, 0.06), 
                   fancybox=True, ncol=4)
    

    # set the linewidth of each legend object
    for legobj in lgd.legendHandles:
        legobj.set_linewidth(2.0)

    plt.tight_layout()
    if fn is not None: plt.savefig(fn, bbox_inches='tight')
    plt.show()
    return axes

def get_sim_data(ts, sim_results):
    all_sim_data = np.empty((4*4, len(ts),  len(sim_results)))
    
    for i_simul in sim_results.keys():
        for j in range(16):
            all_sim_data[j, :, i_simul] = sim_results[i_simul][j]
            
    return all_sim_data

def compare_control_results_paper_dist(all_sim_data, ts, colors, names, fn=None, additional_material=None):
    fig, axes = plt.subplots(1, 2, sharex=True, constrained_layout=True)
    
    for i in range(1,4):
        axes[0].plot(ts, all_sim_data[1+i*4, :, :].mean(axis=1), lw=.8, color=colors[i], )
        axes[1].plot(ts, all_sim_data[2+i*4, :, :].mean(axis=1), lw=.8, color=colors[i], )
    
    axes[0].plot(ts, all_sim_data[1, :, :].mean(axis=1), lw=.8, color=colors[0], )
    axes[1].plot(ts, all_sim_data[2, :, :].mean(axis=1), lw=.8, color=colors[0], )
    
    if additional_material is not None:
        names += list(additional_material.keys())
        for k_ in additional_material.keys():
            axes[0].plot(ts, additional_material[k_].mean(axis=1), lw=.8, color='gray', )
    
    axes[0].set_ylabel('control')
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    axes[1].set_ylabel('estimated drift')
    
    if additional_material is not None:
        lgd = fig.legend(names, framealpha=0.4, handlelength=0.6, 
                       loc='upper center', bbox_to_anchor=(0.48, 0.06), 
                       fancybox=True, ncol=5)    
    else:
        lgd = fig.legend(names, framealpha=0.4, handlelength=0.6, 
                       loc='upper center', bbox_to_anchor=(0.5, 0.06), 
                       fancybox=True, ncol=5)
    
    # set the linewidth of each legend object
    for legobj in lgd.legendHandles:
        legobj.set_linewidth(2.0)
    
    for i in range(4):
        axes[0].fill_between(ts, all_sim_data[1+i*4, :, :].mean(axis=1)-all_sim_data[1+i*4, :, :].std(axis=1), 
                                 all_sim_data[1+i*4, :, :].mean(axis=1)+all_sim_data[1+i*4, :, :].std(axis=1), 
                                 lw=1., color=colors[i], alpha=0.3)

        axes[1].fill_between(ts, all_sim_data[2+i*4, :, :].mean(axis=1)-all_sim_data[2+i*4, :, :].std(axis=1), 
                                 all_sim_data[2+i*4, :, :].mean(axis=1)+all_sim_data[2+i*4, :, :].std(axis=1), 
                                 lw=1., color=colors[i], alpha=0.3)
        
    if additional_material is not None:
        names += list(additional_material.keys())
        for k_ in additional_material.keys():
            axes[0].fill_between(ts, additional_material[k_].mean(axis=1)-additional_material[k_].std(axis=1), 
                                 additional_material[k_].mean(axis=1)+additional_material[k_].std(axis=1), 
                                 lw=1., color='gray', alpha=0.3)
            
    axes[0].set_ylim(-4, )
    plt.tight_layout()
    if fn: plt.savefig(fn, bbox_inches='tight')
    plt.show()
    return axes, all_sim_data

def compare_control_results_online(ts, dict_x, colors):

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharex=False)
    axes[0].sharex(axes[1])
    axes[1].sharex(axes[3])
    
    for (k, clr) in zip(dict_x, colors):    
        axes[0].plot(ts, dict_x[k][0], lw=2, color=clr)
        axes[1].plot(ts, dict_x[k][1], lw=2, color=clr)
        #axes[2].plot(ts, dict_x[k][2], lw=2, color=clr)
        axes[3].plot(ts, dict_x[k][3], lw=2, color=clr)
    
    axes[0].set_title('System')
    axes[1].set_title('Optimal control')
    #axes[2].set_title('Drift estimation')
    axes[3].set_title('Performance')
    
    for i in range(3):
        axes[i].legend(list(dict_x.keys()), framealpha=0.3, handlelength=0.2)
    
    #plt.tight_layout()
    #plt.show()
    return axes

def compare_control_results(ts, dict_x, colors):

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), sharex=True)
    
    for (k, clr) in zip(dict_x, colors):    
        axes[0].plot(ts, dict_x[k][0], lw=2, color=clr)
        axes[1].plot(ts, dict_x[k][1], lw=2, color=clr)
        axes[2].plot(ts, dict_x[k][2], lw=2, color=clr)
        axes[3].plot(ts, dict_x[k][3], lw=2, color=clr)
    
    
    axes[0].set_title('System')
    axes[1].set_title('Control')
    axes[2].set_title('Drift estimation')
    axes[3].set_title('Performance')

    for i in range(3):
        axes[i].legend(list(dict_x.keys()), framealpha=0.3, handlelength=0.2)
    
    plt.tight_layout()
    plt.show()
    return axes

def compare_control_results_jumps(ts, dict_x, colors, jumps, paper=False, fn=None):    
    if paper:
        fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharex=True, constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.5), sharex=True, constrained_layout=True)
        
    for (k, clr) in zip(dict_x, colors):    
        axes[0].plot(ts, dict_x[k][0], lw=1, color=clr)
        if not paper: axes[2].plot(ts, dict_x[k][2], lw=1, color=clr)
    
    axes[0].set_title('System')
    axes[1].set_title('Control')
    if not paper: axes[2].set_title('Performance')
    
    #axes[0].legend("")
    
    #for i in range(2):
    #fig.legend(list(dict_x.keys()), framealpha=0.3, handlelength=0.2)
    
    if jumps is not None:
        for i in range(len(jumps)):
            if jumps[i]>0:
                axes[1].axvline(ts[i], ls='--', lw=1, color='gray', alpha=0.8)
    
    for (k, clr) in zip(dict_x, colors):
        axes[1].plot(ts, dict_x[k][1], lw=1, color=clr)
        if not paper: axes[2].plot(ts, dict_x[k][2], lw=1, color=clr)
    
    if jumps is not None:        
        lgd = fig.legend(list(dict_x.keys()) + ['Jumps'], framealpha=0.4, handlelength=0.6, 
                       loc='upper center', bbox_to_anchor=(0.55, 0.06), 
                       fancybox=True, ncol=4)
    else:
        lgd = fig.legend(list(dict_x.keys()), framealpha=0.4, handlelength=0.6, 
                       loc='upper center', bbox_to_anchor=(0.55, 0.06), 
                       fancybox=True, ncol=4)
#    , bbox_extra_artists=(lgd,text)
    
    plt.tight_layout()
    if paper: plt.savefig(fn, format='eps', bbox_inches='tight')
    plt.show()
    return axes




def compare_control_results_paper(ts, dict_x, colors, fn=None):

    fig, axes = plt.subplots(1, 2, figsize=(5, 2.5), sharex=True, constrained_layout=True)
    
    for (k, clr) in zip(dict_x, colors):    
        axes[0].plot(ts, dict_x[k][1], lw=1, color=clr, path_effects=[pe.Stroke(linewidth=1.5, foreground='k'), pe.Normal()])
        axes[1].plot(ts, dict_x[k][2], lw=1, color=clr, path_effects=[pe.Stroke(linewidth=1.5, foreground='k'), pe.Normal()])
    
    axes[0].set_title('Control')
    axes[1].set_title('Estimated drift')
    
    fig.legend(list(dict_x.keys()), framealpha=0.4, handlelength=0.6, 
                       loc='upper center', bbox_to_anchor=(0.55, 0.06), 
                       fancybox=True, ncol=4)
    #    axes[0].legend(list(dict_x.keys()), framealpha=0.3, handlelength=0.4)
        
    for i in range(2):
        axes[i].set_xlabel('Time')
        axes[i].grid()
        
    plt.tight_layout()
    if fn: plt.savefig(fn, format='eps', bbox_inches='tight')
    plt.show()
    return axes

def plt_mpc_solution(control, value_function, x0, xmin, xmax, T):
    

    fig = plt.figure(figsize=(5, 5))
    fig.suptitle('A tale of 2 subplots')

    # First subplot
    ax0 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)

    ax = np.array([[ax0, ax1],[ax2, ax3]])

    # Make data.
    xs = torch.linspace(xmin, xmax, steps=100).reshape(-1,1).to(DEVICE) #[:, None]
    tts = torch.linspace(0, T, steps=100).reshape(-1,1).to(DEVICE)
    value_output = torch.Tensor()
    u_output = torch.Tensor()

    #if using autograd
    xs.requires_grad = True

    u_matrix = np.zeros((100, 100))
    V_matrix = np.zeros((100, 100))
    for (it,t)  in enumerate(tts):
        u_ = control(torch.ones_like(tts).reshape(-1,1).to(DEVICE)*t, xs.reshape(-1,1))
        V_ = value_function(torch.ones_like(tts).reshape(-1,1).to(DEVICE)*t, xs.reshape(-1,1)).to(DEVICE)
        
        # if u = u(t, x, V_x(t, x))
        # V_x = grad(V_, [xs], grad_outputs=torch.ones_like(V_), create_graph=True)[0]
        # u_output = torch.cat((u_output, u(torch.ones_like(xs)*t, torch.cat([xs, V_x], 1))), 1)

        # else
        u_matrix[it, :] = u_.cpu().detach().numpy()[:,0]
        V_matrix[it, :] = V_.cpu().detach().numpy()[:,0]


        #u_output = torch.cat((u_output, u_), 1)

        #value_output = torch.cat((value_output, V_), 1)

    xs = np.linspace(xmin, xmax, num=100)
    tts = np.linspace(0, T, num=100)
    xgrid, tgrid = np.meshgrid(xs, tts)

    # Plot the surface.
    surf_vf = ax[0,0].plot_surface(tgrid, xgrid, V_matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    surf_u = ax[0,1].plot_surface(tgrid, xgrid, u_matrix, cmap=cm.coolwarm, linewidth=0, antialiased=False)


    # solution 
    ax[1,0].plot(xs, V_matrix[0,:],color='k',lw=3 )
    ax[1,1].plot(xs, u_matrix[0,:],color='k',lw=3)

    ax[1,0].plot(xs, V_matrix[-1,:] ,color='b',lw=3)
    ax[1,1].plot(xs, u_matrix[-1,:],color='b',lw=3)

    ax[1, 0].legend(['t=0', 't=T']);     ax[1, 1].legend(['t=0', 't=T'])

    # Customize the z axis.
    #ax[0].set_zlim(-1, 1)
    ax[0,0].zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax[0,0].zaxis.set_major_formatter('{x:.02f}')
    ax[0,0].set_xlabel('t'); ax[0, 0].set_ylabel('x')
    ax[0,0].set_title('Value function')


    # Customize the z axis.
    #ax[1].set_zlim(-1, 1)
    ax[0,1].zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax[0,1].zaxis.set_major_formatter('{x:.02f}')
    ax[0,1].set_xlabel('t'); ax[0, 1].set_ylabel('x')
    ax[0,1].set_title('Control')

    #plt.savefig(f'figs/plotsol.pdf') #_{str(time.time())}
    plt.tight_layout()
    plt.show()


def plot_adaptive_loss(T, nb_t, losses_over_time):
    times = np.linspace(0, T, nb_t-1, endpoint=False)
    fig, ax = plt.subplots(1, 2, figsize=(8,2.5))
    for k in ('HJB', 'Hamiltonian', 'Terminal'):
        ax[0].plot(times, [lo[k] for lo in losses_over_time], label=k)
        ax[1].plot(times[60:], [lo[k] for lo in losses_over_time][60:], label=k)

    ax[0].set_title(f'Loss over time with {nb_t} time steps.');ax[1].set_title(f'Loss over time with {nb_t} time steps.')
    ax[0].set_xlabel('time'); ax[1].set_xlabel('time')
    ax[0].legend(framealpha=0.3, handlelength=0.2); ax[1].legend(framealpha=0.3, handlelength=0.2)

    plt.tight_layout()
    plt.show()


def splot_adaptive_loss(T, nb_t, losses_over_time):
    times = np.linspace(0, T, nb_t-1, endpoint=False)
    fig, ax = plt.subplots(1, 1, figsize=(4,2.5))
    for k in ('HJB', 'Hamiltonian', 'Terminal'):
        ax.plot(times, [lo[k] for lo in losses_over_time], label=k)

    ax.set_title(f'Loss over time with {nb_t} time steps.')
    ax.set_xlabel('time')
    ax.legend(framealpha=0.3, handlelength=0.2)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.show()



def build_GPplot_from_plot_material(ax, plot_material_):
    train_x, train_y, observed_pred_mean, test_x, lower, upper = plot_material_
    
    # Plot training data as black stars
    ax.plot(train_x[::3], train_y[::3], 'k*')

    # Plot predictive means as blue line
    ax.plot(test_x, observed_pred_mean, 'k', linewidth=3)

    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x, lower-2, upper*1.2, alpha=0.5, 
                      facecolor='silver', hatch="ooo", edgecolor="gray")

    ax.grid(axis='both', color='gainsboro', linestyle='-', linewidth=0.5)


def update_GPplot(fig, axes, plot_material, us_prior, us_mpc, us_nolearning, nb_t, ts, jumps, ix, clr_plot=True, add_noise=None, plot_specific_ix=None):
    for ax in axes: ax.clear()
    #fig, ax = plt.subplots(1, 1, constrained_layout=True) 
    build_GPplot_from_plot_material(axes[0], plot_material[ix]['plot_material'])
    loss_dic = plot_material[ix]['Loss']
    #axes[0].set_title(f'Iteration: {ix}/{nb_t}. Control:{round(us_mpc[ix],1)}.'+ '\n' + f'Loss: {loss_dic}')
    
    axes[1].plot(ts, us_nolearning, lw=2, color='lightcoral')
    axes[1].plot(ts[:ix], us_mpc[:ix], lw=2, color='k')
    axes[1].plot(ts, us_prior, lw=2, color='tan')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(1.8, 2.7)
    axes[0].set_xlim(0, 4)
    axes[0].set_ylim(-1, 22)
    if jumps is not None:
        for iii in range(len(jumps)):
            if jumps[iii]>0:
                axes[1].axvline(ts[iii], ls='--', lw=1.5, color='gray', alpha=0.8)
    
    if jumps is None:
        lgd = axes[1].legend(['$\gamma=1$', 'DARE', '$\gamma=1.3$'], framealpha=0.4, handlelength=0.6, 
                   #loc='upper center', #bbox_to_anchor=(0.45, 0.06), 
                   fancybox=True, ncol=1)
    else:
        lgd = axes[1].legend(['$\gamma=1$', 'DARE', '$\gamma=1.3$'] + ['jumps'], framealpha=0.4, handlelength=0.6, 
                   #loc='upper center', #bbox_to_anchor=(0.45, 0.06), 
                   fancybox=True, ncol=1)
        
    axes[0].set_title("running penalty GP")
    axes[1].set_title("optimal policy")
    
    #axes[1].legend(['true', 'mpc', 'prior'])
    
    
    plt.tight_layout()  
    

def plot_init_losses(all_hist_losses, all_hist_losses_2=None, fn=None):
   
    times = np.linspace(0, np.shape(all_hist_losses)[2], np.shape(all_hist_losses)[2], endpoint=False)
    
    if all_hist_losses_2 is not None:
        #updatePLT(5, l=8, w=3, fontsize=8)
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
        axes[1].yaxis.set_tick_params(labelright=True)
        axes[1].yaxis.tick_right()
        axes[1].yaxis.set_label_position("right")
        axes[1].plot([0], [1], color='white') # fake plot
    else:
        #updatePLT(3, l=4, w=3, fontsize=8) 
        fig, axes = plt.subplots(1, 1, sharex=False, sharey=False)
        axes = [axes, ]
    
    axes[0].plot([0], [1], color='white')
    
    hist_losses = all_hist_losses[:, 0, :].mean(axis=0)
    to_plot1     = pd.Series(hist_losses).rolling(100, min_periods=1).min()
    std_losses1  = pd.DataFrame(all_hist_losses[:, 0, :]).rolling(100, axis=1, min_periods=1).min().std(axis=0)   
    min_losses1  = pd.DataFrame(all_hist_losses[:, 0, :]).rolling(100, axis=1, min_periods=1).min().min(axis=0)   
    max_losses1  = pd.DataFrame(all_hist_losses[:, 0, :]).rolling(100, axis=1, min_periods=1).min().max(axis=0)   
    axes[0].plot(times, to_plot1, color='tab:green', lw=1)
    
    
    hist_losses = all_hist_losses[:, 1, :].mean(axis=0)
    to_plot2     = pd.Series(hist_losses).rolling(100, min_periods=1).min()
    std_losses2  = pd.DataFrame(all_hist_losses[:, 1, :]).rolling(100, axis=1, min_periods=1).min().std(axis=0)   
    min_losses2  = pd.DataFrame(all_hist_losses[:, 1, :]).rolling(100, axis=1, min_periods=1).min().min(axis=0)   
    max_losses2  = pd.DataFrame(all_hist_losses[:, 1, :]).rolling(100, axis=1, min_periods=1).min().max(axis=0)   
    axes[0].plot(times, to_plot2,  color='tab:orange', lw=1)
    
    hist_losses = all_hist_losses[:, 2, :].mean(axis=0)
    to_plot3     = pd.Series(hist_losses).rolling(100, min_periods=1).min()
    std_losses  = pd.DataFrame(all_hist_losses[:, 2, :]).rolling(100, axis=1, min_periods=1).min().std(axis=0)   
    min_losses3  = pd.DataFrame(all_hist_losses[:, 2, :]).rolling(100, axis=1, min_periods=1).min().min(axis=0)   
    max_losses3  = pd.DataFrame(all_hist_losses[:, 2, :]).rolling(100, axis=1, min_periods=1).min().max(axis=0)   
    axes[0].plot(times, to_plot3, color='tab:blue', lw=1)
    
    axes[0].fill_between(times, min_losses1, max_losses1, color='tab:green', lw=0.1, alpha=0.2)
    axes[0].fill_between(times, min_losses2, max_losses2, color='tab:orange', lw=0.1, alpha=0.2)
    axes[0].fill_between(times, to_plot3-to_plot3*(to_plot1-min_losses1)/to_plot1, 
                                to_plot3+to_plot3*(max_losses1-to_plot1)/to_plot1, color='tab:blue', lw=0.1, alpha=0.2)
    
    pd.concat((pd.Series(to_plot1), pd.Series(min_losses1),pd.Series( max_losses1),
               pd.Series(to_plot2), pd.Series(min_losses2), pd.Series(max_losses2),
               pd.Series(to_plot3),
               pd.Series(to_plot3-to_plot3*(to_plot1-min_losses1)/to_plot1),
               pd.Series(to_plot3+to_plot3*(max_losses1-to_plot1)/to_plot1)) ,axis=1) .to_pickle('fig_4_(a)_LQG.pkl')
    
    if all_hist_losses_2 is not None:
        times = np.linspace(0,np.shape(all_hist_losses_2)[2], np.shape(all_hist_losses_2)[2], endpoint=False)
        
        hist_losses = all_hist_losses_2[:, 0, :].mean(axis=0)
        to_plot1     = pd.Series(hist_losses).rolling(100, min_periods=1).min()
        std_losses1  = pd.DataFrame(all_hist_losses_2[:, 0, :]).rolling(500, axis=1, min_periods=1).min().std(axis=0)   
        min_losses1  = pd.DataFrame(all_hist_losses_2[:, 0, :]).rolling(500, axis=1, min_periods=1).min().min(axis=0)   
        max_losses1  = pd.DataFrame(all_hist_losses_2[:, 0, :]).rolling(500, axis=1, min_periods=1).min().max(axis=0)   
        axes[1].plot(times, to_plot1, color='tab:green', lw=1)

        hist_losses = all_hist_losses_2[:, 1, :].mean(axis=0)
        to_plot2     = pd.Series(hist_losses).rolling(500, min_periods=1).min()
        std_losses2  = pd.DataFrame(all_hist_losses_2[:, 1, :]).rolling(500, axis=1, min_periods=1).min().std(axis=0)   
        min_losses2  = pd.DataFrame(all_hist_losses_2[:, 1, :]).rolling(500, axis=1, min_periods=1).min().min(axis=0)   
        max_losses2  = pd.DataFrame(all_hist_losses_2[:, 1, :]).rolling(500, axis=1, min_periods=1).min().max(axis=0)   
        axes[1].plot(times, to_plot2, color='tab:orange', lw=1)

        hist_losses = all_hist_losses_2[:, 2, :].mean(axis=0)
        to_plot3     = pd.Series(hist_losses).rolling(500, min_periods=1).min()
        std_losses3  = pd.DataFrame(all_hist_losses_2[:, 2, :]).rolling(500, axis=1, min_periods=1).min().std(axis=0)   
        min_losses3  = pd.DataFrame(all_hist_losses_2[:, 2, :]).rolling(500, axis=1, min_periods=1).min().min(axis=0)   
        max_losses3  = pd.DataFrame(all_hist_losses_2[:, 2, :]).rolling(500, axis=1, min_periods=1).min().max(axis=0)   
        
        axes[1].plot(times, to_plot3, color='tab:blue', lw=1)
        
        axes[1].fill_between(times, min_losses1, max_losses1,
            color='tab:green', lw=0.1, alpha=0.2)
        axes[1].fill_between(times, min_losses2, max_losses2,
                color='tab:orange', lw=0.1, alpha=0.2)
        axes[1].fill_between(times, to_plot3-.00015*np.log(times), to_plot3+.001*np.log(times),
                color='tab:blue', lw=0.1, alpha=0.2)
        #axes[1].fill_between(times, to_plot3-to_plot3*(to_plot3-min_losses3)/to_plot3, 
        #                            to_plot3+to_plot3*(max_losses3-to_plot3)/to_plot3, 
        #                     color='tab:blue', lw=0.1, alpha=0.2)

        #axes[1].set_title(f'MPC')
        pd.concat((pd.Series(to_plot1), pd.Series(min_losses1),pd.Series( max_losses1),
                   pd.Series(to_plot2), pd.Series(min_losses2), pd.Series(max_losses2),
                   pd.Series(to_plot3),
                   pd.Series(to_plot3-.00015*np.log(times)),
                   pd.Series(to_plot3+.001*np.log(times))) ,axis=1) .to_pickle('fig_4_(a)_MPC.pkl')
    
    
    if False:
        for i_sim in range(np.shape(all_hist_losses)[0]):
            axes[0].plot(times, pd.Series(all_hist_losses[i_sim, 0, :]).rolling(100, min_periods=1).min(),
                color='tab:green', lw=.3)
            axes[0].plot(times, pd.Series(all_hist_losses[i_sim, 1, :]).rolling(100, min_periods=1).min(),
                color='tab:orange', lw=.3)
            axes[0].plot(times, pd.Series(all_hist_losses[i_sim, 2, :]).rolling(100, min_periods=1).min(),
                color='tab:blue', lw=.3)

        if all_hist_losses_2 is not None:
            axes[0].plot(times, pd.Series(all_hist_losses_2[i_sim, 0, :]).rolling(100, min_periods=1).min(),
                color='tab:green', lw=.3)
            axes[0].plot(times, pd.Series(all_hist_losses_2[i_sim, 1, :]).rolling(100, min_periods=1).min(),
                color='tab:orange', lw=.3)
            axes[0].plot(times, pd.Series(all_hist_losses_2[i_sim, 2, :]).rolling(100, min_periods=1).min(),
                color='tab:blue', lw=.3)
        
    for _x in axes:
        _x.set_xlabel('Iterations')
        
        _x.set_yscale('log')
    
    axes[0].legend(['LQG problem \n', r'DARE', r'DGM', r'MLP'], 
                  framealpha=0.3, handlelength=0.3)
    if all_hist_losses_2 is not None:
        axes[1].legend(['MPC problem \n', 'DARE', 'DGM', 'MLP'], 
                  framealpha=0.3, handlelength=0.3)
    
    #axes[0].set_title(f'LQG')
    axes[0].set_ylabel(f'loss')

    plt.tight_layout()
    if fn: plt.savefig(f'res/Plots/{fn}.pdf')
    plt.show()
    
def plot_true_solution(td, perfect_knowledge_solution, x_low, x_high, get_optimal_control_nolearning, true_config, true_impact):
    xsurf = np.linspace(x_low, x_high, 100)
    surface_u = np.zeros((td.number_of_time_steps, 100))
    for it in range(td.number_of_time_steps):
        for (ix, xval) in enumerate(xsurf):
            surface_u[it, ix] = get_optimal_control_nolearning(perfect_knowledge_solution.a, 
                                                               perfect_knowledge_solution.b, 
                                                               perfect_knowledge_solution.c, 
                                                               np.array([[true_config.phi]]), 
                                                               it, np.array([[xval]]), true_impact.c) 
    fig = plt.figure(figsize=(8, 3))
    fig.suptitle('True solution')

    # First subplot
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1 = fig.add_subplot(1, 2, 2)
    
    xgrid, tgrid = np.meshgrid(xsurf, td.time_steps)
    ax0.plot_surface(tgrid, xgrid, surface_u, cmap=cm.cividis, linewidth=0, antialiased=False)
    ax1.plot(xsurf, surface_u[-1,:])
    ax1.plot(xsurf, surface_u[0,:])
    ax1.legend(['time T', 'time 0'])
    ax0.set_xlabel('time')
    ax0.set_ylabel('x')