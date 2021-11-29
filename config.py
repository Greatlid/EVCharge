''' Constants for the algorithms '''

N_EV = 50
TIME_HORIZON = 24 #total time interval number
MaxCapacities = 80  #max capacities:80 kwh
MaxPower = 80 #max power:80 kw/h
Efficiencies = 1 #charge efficiency
N_ChargeStation = 4 #
N_ChargeUnit = 10
Time_Value = 20
MaxIteration = 15
ladder_price = [0.4983, 0.5483, 0.7983]
ladder_power = [0, 2800, 3200]
price_ub = 1.5
price_lb = 0.5
price_mean = 1
Y = 7  #User state transition penalty factor
alpha = 1
beta = 5
N_node = 45
charge_station_index = [t-1 for t in [7, 12, 33, 36]]
v_car = 20
weight_time = 0.5
weight_cs = 0.5