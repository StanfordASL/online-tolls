import os
import numpy as np
import pandas as pd
from datetime import datetime

from network import Network
from users import Users
from user_equilibrium_with_tolls import UserEquilibriumWithTolls
from optimal_flow import OptimalFlow


class IidOdExpts:

    def __init__(self):
        # Create a folder for this experiment
        self.folder_path = self.create_folder()

        # city name for the experiment
        self.city = 'SiouxFalls'

        # Range of T values for the experiment
        self.Trange = [5, 25, 50, 100, 250, 500, 1000]
        # self.Trange = [2, 5]

        # users for the experiment
        users = Users(self.city)

        # city network for the experiment
        network = Network(self.city)
        self.num_physical_edges = network.NumEdges
        network.add_outside_option(users)  # Adding the outside option links

        self.compare(network, users)

    def compare(self, network, users):

        log = pd.DataFrame(columns=['T', 'regret_gr_desc_not_normalized', 'vio_gr_desc_not_normalized'])

        for T in self.Trange:

            # Initialize performance parameters
            obj_gr_desc = 0
            obj_opt = 0
            vio_gr_desc = np.zeros(network.NumEdges)

            # Initialize gr_desc tolls
            gr_desc_tolls = np.zeros(network.NumEdges)

            step_size = 5e-4 / np.sqrt(T)

            for t in range(T):
                print("[IidOdExpts] Iteration %d of %d" % (t, T))

                # compute optimal flow
                opt_solver = OptimalFlow(network, users)
                opt_solver.set_obj(users)
                x_opt, f_opt = opt_solver.solve()

                obj_opt += network.latency_array() @ x_opt @ users.vot_array()

                # Gradient descent algorithm

                ue_with_tolls = UserEquilibriumWithTolls(network, users, gr_desc_tolls)
                ue_with_tolls.set_obj(users, gr_desc_tolls)
                x, f = ue_with_tolls.solve()

                obj_gr_desc += network.latency_array() @ x @ users.vot_array()
                vio_gr_desc += f - network.capacity

                # Updating gradient descent tolls
                gr_desc_tolls += step_size * (f - np.array(network.capacity))
                gr_desc_tolls[gr_desc_tolls < 0] = 0

                # Draw a new user VOT realization for next time step
                users.new_instance()

            # Parameter logging
            log = log.append({'T': T,
                              'regret_gr_desc_not_normalized': obj_gr_desc - obj_opt,
                              'vio_gr_desc_not_normalized': max(vio_gr_desc)},
                             ignore_index=True)
            log.to_csv(self.folder_path + 'iid_comparison.csv')

    @staticmethod
    def create_folder():
        root_folder_path = 'Results/'
        if os.path.isdir(root_folder_path):
            pass
        else:
            os.mkdir(root_folder_path)
        return root_folder_path
