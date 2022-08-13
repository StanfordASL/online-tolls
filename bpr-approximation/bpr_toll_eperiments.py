from network import Network
from users import Users
from user_equilibrium_with_tolls import UserEquilibriumWithTolls
from optimal_flow import OptimalFlow
import pandas as pd
import numpy as np
from utils import *


class BPRTollExperiments:

    def __init__(self, road_network=None):

        # city name for the experiment
        self.city = road_network

        # Create a folder for this experiment
        self.folder_path = self.create_folder(subpath=road_network)

        # Range of T values for the experiment
        # self.Trange = [5, 25, 50, 100, 250, 500, 1000, 5000, 10000, 20000, 50000, 100000]
        self.Trange = [100, 500, 1000, 2000, 5000, 10000]

        # users for the experiment
        users = Users(self.city)

        # city network for the experiment
        network = Network(self.city)
        self.num_physical_edges = network.NumEdges

        # evaluate the BPR codes
        self.simulate_bpr(network, users)

    @staticmethod
    def bpr(f, capacity, free_flow_latency):
        """
        :param f: edge flow numpy array
        :param capacity: edge capacity array
        :param free_flow_latency: free flow latency
        :return: numpy array of travel times on edges
        """
        if len(f) != len(capacity):
            print("[bpr-function] Error!")

        edge_tt = np.zeros(len(f))

        for e in range(len(edge_tt)):
            if f[e] < 1.1 * capacity[e]:
                edge_tt[e] = 1.025 * free_flow_latency[e]
            else:
                edge_tt[e] = 1.025 * free_flow_latency[e] + 0.6 * (f[e] - 1.1 * capacity[e])
        return edge_tt

    def simulate_bpr(self, network, users):

        # We need to compare the performance of different values of x_tar for the same VOT realizations

        log = pd.DataFrame(columns=['T', 'regret_opt_target', 'average_opt_target', 'normalized_opt_target',
                                    'regret_congestion_target', 'average_congestion_target',
                                    'normalized_congestion_target'])

        # Initializing user equilibrium solver
        ue_with_tolls = UserEquilibriumWithTolls(network, users, np.zeros(network.NumEdges), piecewise_bpr=True)

        # initializing optimal flow solver
        opt_solver = OptimalFlow(network, users, piecewise_bpr=True)

        # Initialize the target flow
        # Users have mean VOTs
        users.new_instance(fixed_vot=True)
        opt_solver.set_obj(users)
        _, f_target = opt_solver.solve()

        for T in self.Trange:
            cum_regret_opt_target = 0
            cum_regret_congestion_target = 0
            cumulative_so = 0

            step_size = 1e0 / np.sqrt(T)

            # initializing gradient descent tolls
            tolls_opt_target = np.zeros(network.NumEdges)
            tolls_congestion_target = np.zeros(network.NumEdges)

            print("[OnlineTollExperiments] Solving for T = ", T)

            for t in range(T):
                # print("[OnlineTollExperiments] Iteration %d of %d" % (t, T))

                # system optimal flows
                opt_solver.set_obj(users)
                x, f = opt_solver.solve()

                edge_tt = self.bpr(f, network.capacity_array(), network.latency_array())
                so_obj = edge_tt @ x @ users.vot_array()
                cumulative_so += so_obj

                # x_{Tar} is the optimal flow:
                ue_with_tolls.set_obj(users, tolls_opt_target)
                x, f = ue_with_tolls.solve()  # x[e,u] f[e]
                edge_tt = self.bpr(f, network.capacity_array(), network.latency_array())
                gr_desc_obj = edge_tt @ x @ users.vot_array()
                tolls_opt_target += step_size * (f - f_target)
                tolls_opt_target[tolls_opt_target < 0] = 0
                # updating regret for target = optimal flow
                cum_regret_opt_target += gr_desc_obj - so_obj

                # x_{Tar} is the flow at the congestion limit of 1.1 * capacity:
                ue_with_tolls.set_obj(users, tolls_congestion_target)
                x, f = ue_with_tolls.solve()  # x[e,u] f[e]
                edge_tt = self.bpr(f, network.capacity_array(), network.latency_array())
                gr_desc_obj = edge_tt @ x @ users.vot_array()
                tolls_congestion_target += step_size * (f - 1.1 * np.array(network.capacity))
                tolls_congestion_target[tolls_congestion_target < 0] = 0
                # updating regret
                cum_regret_congestion_target += gr_desc_obj - so_obj

                # Draw a new user VOT realization for next time step
                users.new_instance()

            log = log.append({'T': T,
                              'regret_opt_target': cum_regret_opt_target,
                              'average_opt_target': cum_regret_opt_target / T,
                              'normalized_opt_target': cum_regret_opt_target / cumulative_so,
                              'regret_congestion_target': cum_regret_congestion_target,
                              'average_congestion_target': cum_regret_congestion_target / T,
                              'normalized_congestion_target': cum_regret_congestion_target /cumulative_so},
                             ignore_index=True)

            log.to_csv(self.folder_path + 'log.csv')

        return None

    @staticmethod
    def create_folder(subpath=None):
        root_folder_path = 'Results/'
        if os.path.isdir(root_folder_path):
            pass
        else:
            os.mkdir(root_folder_path)
        if subpath is None:
            return root_folder_path
        else:
            if os.path.isdir(root_folder_path + subpath):
                pass
            else:
                os.mkdir(root_folder_path + subpath)
            return root_folder_path + subpath + '/'

