from bpr_toll_eperiments import BPRTollExperiments

# Two-Edge Parallel Network
BPRTollExperiments(road_network='Pigou_bpr')

# Six-Edge parallel Network
BPRTollExperiments(road_network='Pigou_expanded')

# Series-Parallel Network
BPRTollExperiments(road_network='Series_parallel')

# Grid Network
BPRTollExperiments(road_network='Grid')
