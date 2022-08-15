# online-tolls

This repository contains code associated with a submission to [AAAI 2023](https://aaai.org/Conferences/AAAI-23/). 

## Data sources ##
The traffic network, user flows, and road capacities are obtained from the [TNTP dataset](https://github.com/bstabler/TransportationNetworks). The Sioux Falls dataset is present in the `Locations` folder. Test networks for the congestion games section (Appendix G) is present in the [bpr-approximation/Locations](bpr-approximation/Locations) folder

## Requirements ##

This code uses the following packages
- [Gurobi](https://www.gurobi.com/products/gurobi-optimizer/) for the optimization solvers
- [Geopandas](https://geopandas.org/en/stable/) for manipulating geospatial data
- [Contextily](https://contextily.readthedocs.io/en/latest/) for loading basemap for plots

See [requirements.txt](requirements.txt) for a complete list

## Running the code ##

- [main.py](main.py) to run simulations
- [plots.py](plots.py) to generate the plots 
- [bpr-approximation/](bpr-approximation/) contins the experiments on congetion games (Appendix G)
    - [bpr-approximation/main.py](bpr-approximation/main.py) runs the simulations
    - [bpr-approximation/plots_bpr.py](bpr-approximation/plots_bpr.py) generates the plots
    - [bpr-approximation/bpr_approximation.ipynb](bpr-approximation/bpr_approximation.ipynb) generates plots to validate the piecewise-lienar approximation
