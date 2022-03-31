
import random
import pandas as pd
import geopandas as gpd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import contextily as cx
from utils import *


data_path = 'Results/'
fig_path = 'Results/Figures/'

if os.path.isdir(fig_path):
    pass
else:
    os.mkdir(fig_path)


"""
####################
Average Normalized Regret and Capacity Violation
####################
"""

df = pd.read_csv(data_path + 'comparison.csv')

plt.rcParams['font.size'] = '14'
fig, axes = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(12, 6)

axes[0].plot(df['T'], df['regret_gr_desc'], '*-', c='tab:blue', label='Algorithm 1')
axes[0].plot(df['T'], df['regret_stochastic'], '*-', c='tab:orange', label='User mean VoT')
axes[0].plot(df['T'], df['regret_no_vot'], '*-', c='tab:green', label='Population mean VoT')
axes[0].plot(df['T'], df['regret_const_update'], '*-', c='tab:red', label='Reactive update')
axes[0].set_xlabel('Time Periods')
axes[0].set_ylabel('Average Normalized Regret')
axes[0].legend(loc="lower right")

axes[1].plot(df['T'], df['vio_gr_desc'], '*-', c='tab:blue', label='Algorithm 1')
axes[1].plot(df['T'], df['vio_stochastic'], '*-', c='tab:orange', label='User mean VoT')
axes[1].plot(df['T'], df['vio_no_vot'], '*-', c='tab:green', label='Population mean VoT')
axes[1].plot(df['T'], df['vio_const_update'], '*-', c='tab:red', label='Reactive update')
axes[1].set_xlabel('Time Periods')
axes[1].set_ylabel('Average Normalized Capacity Violation')
axes[1].legend(loc="upper right")


plt.tight_layout()
plt.savefig(fig_path + 'sqrtTconvergence.png', dpi=250)

plt.close()


"""
####################
Average normalized Travel time 
####################
"""

df = pd.read_csv(data_path + 'comparison.csv')

plt.rcParams['font.size'] = '14'

fig, axes = plt.subplots(nrows=1, ncols=1)
fig.set_size_inches(6, 6)

axes.plot(df['T'], 1 + df['ttt_gr_desc'], '*-', c='tab:blue', label='Algorithm 1')
axes.plot(df['T'], 1 + df['ttt_stochastic'], '*-', c='tab:orange', label='User mean VoT')
axes.plot(df['T'], 1 + df['ttt_no_vot'], '*-', c='tab:green', label='Population mean VoT')
axes.plot(df['T'], 1 + df['ttt_const_update'], '*-', c='tab:red', label='Reactive update')
axes.set_xlabel('Time Periods')
# axes.set_ylabel('$\\dfrac{ \\mathrm{Total \;\, Travel \;\, Time}}{ \\mathrm{Optimal \;\, Total \;\, Travel \;\, '
#                 'Time}}$')
axes.set_ylabel('Average Normalized Total Travel Time')
axes.legend(loc="lower right")

plt.tight_layout()
plt.savefig(fig_path + 'totaltraveltime.png', dpi=250)

plt.close()


"""
####################
Map plots of networks: Topology, capacity, latency
####################
"""


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_edge_values(value, figpath, truncate_flag=True, plot_lim=None, label=None, text=None):

    # Load vertices
    df_vertices = pd.read_csv('Locations/SiouxFalls/vertices.csv')

    # Load edges
    df_edges = pd.read_csv('Locations/SiouxFalls/edges.csv')

    # Initialize figure
    plt.rcParams['font.size'] = '22'
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 10)

    # Create Nodes for Plotting
    gdf_vertices = gpd.GeoDataFrame(
        df_vertices, geometry=gpd.points_from_xy(df_vertices.xcoord, df_vertices.ycoord), crs="EPSG:4326")
    gdf_vertices = gdf_vertices.to_crs('EPSG:3857')

    # Plot Nodes
    ax = gdf_vertices.plot(color='red', ax=ax)

    # Add basemap
    cx.add_basemap(ax, alpha=0.5)

    # Create edges
    gdf_vertices['lat'] = gdf_vertices.geometry.y
    gdf_vertices['lon'] = gdf_vertices.geometry.x

    merge_df = gdf_vertices.filter(['vert_id', 'lat', 'lon'], axis=1)
    merge_df = merge_df.rename(columns={'lat': 'tail_lat', 'lon': 'tail_lon'})

    df_edges = df_edges.merge(merge_df, how='left', left_on='edge_tail', right_on='vert_id')
    df_edges = df_edges.drop(columns=['vert_id'])

    merge_df = merge_df.rename(columns={'tail_lat': 'head_lat', 'tail_lon': 'head_lon'})
    df_edges = df_edges.merge(merge_df, how='left', left_on='edge_head', right_on='vert_id')
    df_edges = df_edges.drop(columns=['vert_id'])

    # set colormap
    cm = plt.get_cmap('Oranges')
    if truncate_flag:
        cm = truncate_colormap(cm, minval=0.3, maxval=1, n=100)
    if plot_lim is not None:
        c_norm = colors.Normalize(vmin=plot_lim[0], vmax=plot_lim[1])
    else:
        c_norm = colors.Normalize(vmin=min(value), vmax=max(value))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=cm)

    df_edges = df_edges.reset_index()  # make sure indexes pair with number of rows

    for index, row in df_edges.iterrows():
        colorval = scalar_map.to_rgba(value[index])
        a = patches.FancyArrowPatch((row['tail_lon'], row['tail_lat']),
                                    (row['head_lon'], row['head_lat']),
                                    connectionstyle="arc3,rad=.1",
                                    arrowstyle="Fancy, head_length=7, head_width=5",
                                    color=colorval)
        # Adding each edge
        plt.gca().add_patch(a)

    # Adding colorbar
    plt.colorbar(scalar_map, ax=ax)
    plt.title(label)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    if text is not None:
        x_pos = -1.0775e7
        y_pos = 5.405e6
        for line in text:
            # print('adding text')
            plt.text(x_pos, y_pos, line, weight="bold")
            y_pos -= 1e3

    # plt.tight_layout()
    fig.savefig(figpath, dpi=250)

    plt.close()


def plot_topology(figpath):

    # Load vertices
    df_vertices = pd.read_csv('Locations/SiouxFalls/vertices.csv')

    # Load edges
    df_edges = pd.read_csv('Locations/SiouxFalls/edges.csv')

    # Initialize figure
    plt.rcParams['font.size'] = '22'
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 10)

    # Create Nodes for Plotting
    gdf_vertices = gpd.GeoDataFrame(
        df_vertices, geometry=gpd.points_from_xy(df_vertices.xcoord, df_vertices.ycoord), crs="EPSG:4326")
    gdf_vertices = gdf_vertices.to_crs('EPSG:3857')

    # Plot Nodes
    ax = gdf_vertices.plot(color='navy', ax=ax)

    # Add basemap
    cx.add_basemap(ax, alpha=0.5)

    # Create edges
    gdf_vertices['lat'] = gdf_vertices.geometry.y
    gdf_vertices['lon'] = gdf_vertices.geometry.x

    merge_df = gdf_vertices.filter(['vert_id', 'lat', 'lon'], axis=1)
    merge_df = merge_df.rename(columns={'lat': 'tail_lat', 'lon': 'tail_lon'})

    df_edges = df_edges.merge(merge_df, how='left', left_on='edge_tail', right_on='vert_id')
    df_edges = df_edges.drop(columns=['vert_id'])

    merge_df = merge_df.rename(columns={'tail_lat': 'head_lat', 'tail_lon': 'head_lon'})
    df_edges = df_edges.merge(merge_df, how='left', left_on='edge_head', right_on='vert_id')
    df_edges = df_edges.drop(columns=['vert_id'])

    for index, row in df_edges.iterrows():
        a = patches.FancyArrowPatch((row['tail_lon'], row['tail_lat']),
                                    (row['head_lon'], row['head_lat']),
                                    connectionstyle="arc3,rad=.1",
                                    arrowstyle="Fancy, head_length=7, head_width=5",
                                    color='black')
        # Adding each edge
        plt.gca().add_patch(a)

    # Turn off tick labels
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # plt.tight_layout()
    fig.savefig(figpath, dpi=250)

    plt.close()


# load data
df_capacity = pd.read_csv(data_path + 'capacity.csv', header=None)
df_latency = pd.read_csv(data_path + 'latency.csv', header=None)
capacity = (df_capacity[0]/2.4).to_list()  # for a per hour number
latency = (df_latency[0]*60).to_list()  # for units in minutes


# make plots
plot_edge_values(capacity, fig_path + 'capacity.png', truncate_flag=True,
                 label='Capacity (vehicles/hour)')
plot_edge_values(latency, fig_path + 'latency.png', truncate_flag=True,
                 label='Travel time (minutes)')
plot_topology(fig_path + 'topology.png')


"""
####################
Plotting tolls on the map:
    - Gradient descent tolls
    - Constant update (i.e., reactive tolling)
    - Population mean Vot
    - User mean VoT
####################
"""

'''

Need to plot the following files on the map:

tolls_gr_desc_t_1000.csv
tolls_const_update_t_1000.csv
group_specific_VOT_toll.csv
population_mean_toll.csv

'''

# load data
gr_desc_toll = pd.read_csv(data_path + 'tolls_gr_desc_t_1000.csv', header=None)
gr_desc_toll = gr_desc_toll[0].to_list()

const_update_toll = pd.read_csv(data_path + 'tolls_const_update_t_1000.csv', header=None)
const_update_toll = const_update_toll[0].to_list()

group_specific_toll = pd.read_csv(data_path + 'group_specific_VOT_toll.csv', header=None)
group_specific_toll = group_specific_toll[0].to_list()

population_mean_toll = pd.read_csv(data_path + 'population_mean_toll.csv', header=None)
population_mean_toll = population_mean_toll[0].to_list()


plot_edge_values(gr_desc_toll, fig_path + 'toll_gr_desc.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.77',
                       'Max. Toll: 2.54'])

plot_edge_values(const_update_toll, fig_path + 'toll_reactive_update.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.73',
                       'Max. Toll: 2.40'])

plot_edge_values(group_specific_toll, fig_path + 'toll_group_vot.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.80',
                       'Max. Toll: 2.63'])

plot_edge_values(population_mean_toll, fig_path + 'toll_pop_vot.png',
                 truncate_flag=False,
                 label='Toll (dollars)',
                 plot_lim=[0, 2.7],
                 text=['Avg. Toll: 0.45',
                       'Max. Toll: 2.71'])

"""
Push relevant statistics from the map into a table
"""
table_path = fig_path + 'stats_table.csv'
try:
    os.remove(table_path)
except:
    pass

write_row(table_path, ['Algorithm', 'MinToll', 'MaxToll', 'NumOfEdgesWithTolls', 'AverageToll'])

non_zero_tolls = [toll for toll in gr_desc_toll if toll > 1e-2 ]
write_row(table_path, ['Gradient descent',
                       min(gr_desc_toll),
                       max(gr_desc_toll),
                       sum(np.array(gr_desc_toll) > 1e-2),
                       np.mean(non_zero_tolls)])


non_zero_tolls = [toll for toll in const_update_toll if toll > 1e-2 ]
write_row(table_path, ['Reactive update',
                       min(const_update_toll),
                       max(const_update_toll),
                       sum(np.array(const_update_toll) > 1e-2),
                       np.mean(non_zero_tolls)])

non_zero_tolls = [toll for toll in group_specific_toll if toll > 1e-2]
write_row(table_path, ['Group Specific VoT',
                       min(group_specific_toll),
                       max(group_specific_toll),
                       sum(np.array(group_specific_toll) > 1e-2),
                       np.mean(non_zero_tolls)])

non_zero_tolls = [toll for toll in population_mean_toll if toll > 1e-2]
write_row(table_path, ['Population VoT',
                       min(population_mean_toll),
                       max(population_mean_toll),
                       sum(np.array(population_mean_toll) > 1e-2),
                       np.mean(population_mean_toll)])

"""
####################
Histogram of tolls generated by our algorithm
####################
"""
gr_desc_toll = pd.read_csv(data_path + 'tolls_gr_desc_t_1000.csv', header=None)
gr_desc_toll = gr_desc_toll[0].to_list()

nbin = 20
count, bins = np.histogram(gr_desc_toll, bins=nbin)

xloc = [(bins[i] + bins[i+1])/2 for i in range(len(bins) -1)]

plt.bar(xloc, count/sum(count)*100, width=bins[1]-bins[0])
plt.xlabel('Tolls (dollars)')
plt.ylabel('Percentage of edges')

plt.tight_layout()
plt.savefig(fig_path + 'tolls_histogram.png')
plt.close()


"""
####################
Convergence of tolls within a few iterations
####################
"""

t1000_path = data_path + 'T_1000_log.csv'

df = pd.read_csv(t1000_path)

plt.rcParams['font.size'] = '14'
plt.plot(df['t'], df[' total_const_update'], label='Reactive update', alpha=0.7)
plt.plot(df['t'], df[' total_gr_desc'], label='Algorithm 1', alpha=0.7)
plt.legend(loc='lower right')
plt.xlabel('Time Periods')
plt.ylabel('Total tolls (dollars)')

plt.savefig(fig_path + 'toll_convergence.png')


"""
#################################
Performance plot (merge regret, capacity violation and total travel time) 
##################################
"""

df = pd.read_csv(data_path + 'comparison.csv')

plt.rcParams['font.size'] = '18'

fig, axes = plt.subplots(nrows=1, ncols=3)
fig.set_size_inches(18, 6)

axes[0].plot(df['T'], df['regret_gr_desc'], '*-', c='tab:blue', label='Algorithm 1')
axes[0].plot(df['T'], df['regret_stochastic'], '*-', c='tab:orange', label='User mean VoT')
axes[0].plot(df['T'], df['regret_no_vot'], '*-', c='tab:green', label='Population mean VoT')
axes[0].plot(df['T'], df['regret_const_update'], '*-', c='tab:red', label='Reactive update')
axes[0].set_xlabel('Time Periods')
axes[0].set_ylabel('Average Normalized Regret')
axes[0].legend(loc="lower right")

axes[1].plot(df['T'], df['vio_gr_desc'], '*-', c='tab:blue', label='Algorithm 1')
axes[1].plot(df['T'], df['vio_stochastic'], '*-', c='tab:orange', label='User mean VoT')
axes[1].plot(df['T'], df['vio_no_vot'], '*-', c='tab:green', label='Population mean VoT')
axes[1].plot(df['T'], df['vio_const_update'], '*-', c='tab:red', label='Reactive update')
axes[1].set_xlabel('Time Periods')
axes[1].set_ylabel('Average Normalized Capacity Violation')
# axes[1].legend(loc="upper right")

axes[2].plot(df['T'], 1 + df['ttt_gr_desc'], '*-', c='tab:blue', label='Algorithm 1')
axes[2].plot(df['T'], 1 + df['ttt_stochastic'], '*-', c='tab:orange', label='User mean VoT')
axes[2].plot(df['T'], 1 + df['ttt_no_vot'], '*-', c='tab:green', label='Population mean VoT')
axes[2].plot(df['T'], 1 + df['ttt_const_update'], '*-', c='tab:red', label='Reactive update')
axes[2].set_xlabel('Time Periods')
axes[2].set_ylabel('Average Normalized Total Travel Time')
# axes[2].legend(loc="lower right")

plt.tight_layout()
plt.savefig(fig_path + 'performance_plots_merged.png', dpi=250)

plt.close()


""" 
##########
Plot for IID OD pairs
#########
"""

df = pd.read_csv(data_path + 'iid_comparison.csv')

# Capacity violation plots
plt.rcParams['font.size'] = '24'

plt.figure(figsize=(8, 6), dpi=250)

x = np.log(np.array(df['T']))
y = np.log(np.array(df['vio_gr_desc_not_normalized']))

# If you want a linear fit
# lin_fit = linregress(x[:], y[:])
# y_lin = lin_fit.slope * x[:] + lin_fit.intercept
# plt.plot(x[:], y_lin, '--', linewidth=3, label='Linear fit')

# If you want the best fit with slope 0.5, do this instead:
bias_vec = y[:] - 0.5 * x[:]
bias = np.mean(bias_vec)
y_lin = 0.5 * x[:] + bias
plt.plot(x[:], y_lin, '--', linewidth=5, color='C1', label='Theoretical bound')


# plt.text(3.5, 9.4, 'R-square = %.3f' % (lin_fit.rvalue**2))
# plt.text(3.5, 9.1, 'Slope = %.3f' % lin_fit.slope)
rmse = np.sqrt(np.mean((y - y_lin)**2))
# plt.text(3.5, 9.1, 'RMSE = %.3f' % rmse)

plt.plot(x, y, 's', linewidth=3, markersize=10, markeredgecolor='black', markerfacecolor='black', label='Algorithm 1')
plt.xlabel('log(Time Periods)')
plt.ylabel('log(Capacity Violation)')


plt.legend(frameon=False, loc='upper left')
plt.tight_layout()
plt.savefig(fig_path + 'iid_cap_vio.png')
plt.close()

# regret plot
x = np.array(df['T'])
y = np.array(df['regret_gr_desc_not_normalized'])

plt.figure(figsize=(8, 6.3), dpi=250)
plt.plot(x, y, 's', linewidth=3, markersize=10, markeredgecolor='black', markerfacecolor='black', label='Algorithm 1')
plt.xlabel('Time Periods')
plt.ylabel('Regret')

# plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(fig_path + 'iid_regret.png')
plt.close()

