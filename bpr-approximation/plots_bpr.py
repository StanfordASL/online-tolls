import pandas as pd
import matplotlib.pyplot as plt
from utils import *


def plot_regret(networkname=None):
    """
    Plot normalized regret trend for a given network file
    """
    network_name = networkname

    # create folders
    path = 'Results/' + network_name + '/'
    fig_path = 'Results/Figures/'

    if os.path.isdir(fig_path):
        pass
    else:
        os.mkdir(fig_path)

    # read data file
    df = pd.read_csv(path + 'log.csv')

    # figure parameters
    plt.rcParams['font.size'] = '16'
    plt.figure(figsize=(7, 5), dpi=250)
    ax = plt.gca()

    #  first lines
    plt.plot(df['T'], df['normalized_opt_target'], '*-', c='tab:blue',
                 label="$\mathbf{x}^{Tar} = \overline{\mathbf{x}}$")
    plt.plot(df['T'], df['normalized_congestion_target'], '*-', c='tab:red',
                 label="$\mathbf{x}^{Tar} = 1.1 \mathbf{c}$")

    # labels
    ax.set_xlabel('Time Periods')
    ax.set_ylabel('Average Normalized Regret')
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()
    plt.savefig(fig_path + networkname + '_regret.png', dpi=250)
    plt.close()

    return None


# Generating Figures in Appendix G
plot_regret(networkname='Pigou_bpr')
plot_regret(networkname='Pigou_expanded')
plot_regret(networkname='Series_parallel')
plot_regret(networkname='Grid')
