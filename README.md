# online-tolls

This repository contains code associated with the paper [Online Learning for Traffic Routing under Unknown Preferences] by Devansh Jalota, Karthik Gopalakrishhah, Navid Azizan, Ramesh Johari, and Marco Pavone.

## Data sources ##
The traffic network, user flows, and road capacities are obtained from the [TNTP dataset](https://github.com/bstabler/TransportationNetworks)

## Requirements ##

This code uses the following packages
- [Gurobi](https://www.gurobi.com/products/gurobi-optimizer/) for the optimization solvers
- [Geopandas](https://geopandas.org/en/stable/) for manipulating geospatial data
- [Contextily](https://contextily.readthedocs.io/en/latest/) for loading basemap for plots

## Running the code ##

- `main.py` to run simulations
- `plots.py` to generate the plots 
