## this script runs the hyperparam search on the toy problem
python param_search_toy.py configs/toy/CEM.json 4
python param_search_toy.py configs/toy/CEM-E.json 16
python param_search_toy.py configs/toy/CEM-GMM.json 64

## the output of the above is the logs for each hyperparam.
## to find the best hyperparam, we first organize manually the logs by the population size,
## e.g.
##   --CEM-E:
##      |-- pop100
##      |-- pop200
##      |-- pop500
##          ...
## then we run the function plot_1d1 in the utils/plotter.py script to 
##  find the best hyperparam and plot their results
