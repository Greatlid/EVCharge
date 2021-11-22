from data_generate import *
from config import *
import numpy as np
import pandas as pd
from cvxpy import Variable as V, Problem as PB, sum_squares as SS, Minimize as MIN, sum as SM
import random
import matplotlib.pyplot as plt

class OffLineModel():
    def __init__(self):
        self.base_load = np.loadtxt('./data/load_data.txt')[:,1]
        self.df_ev = pd.read_csv('./data/ev.csv', sep=',')
        # SOC of EVs by arrival
        self.soc_arr = np.array(self.df_ev['SOC at arrival'])
        # Desired SOC of EVs by departure
        self.soc_dep = np.array(self.df_ev['SOC at departure'])
        # Battery capacity of EVs (in kWh)
        self.cap = np.array(self.df_ev['Battery capacity'])
        # Amount of power required by EVs (in kWh)
        self.power_req = self.cap * (self.soc_dep - self.soc_arr)
        # Maximum charging rate of EVs (in kW)
        self.p_max = np.array(self.df_ev['Maximum power'])
        # Plug-in time of EVs
        self.t_plug_in = np.array(self.df_ev['Plug-in time'].astype(int))
        # Plug-out time of EVs
        self.t_plug_out = np.array(self.df_ev['Plug-out time'].astype(int))
        # Charging efficiency of EVs
        self.efficiency = np.array(self.df_ev['Charging efficiency'])
        #Charging time
        self.t_charge = np.ceil(self.power_req/self.p_max)
        #price adjustment constant
        self.Y = random.uniform(0, (1 / N_EV))
        # Check feasibility of charging
        for n in range(N_EV):
            if (self.t_plug_out[n] - self.t_plug_in[n]) < (self.power_req[n] / self.p_max[n]):
                raise ValueError('Solution is not feasible')

    def calculate_price_signal(self, base_load, charging_schedules):
        """ Calculate the price signal based on updated charging schedules of EVs.
        Price during time t is modeled as a function of total demand during time t
        Keyword arguments:
            base_load : Non-EV load
            charging_schedule : Updated charging schedules of EVs
        Returns:
            new_price : Control signal (price) for next iteration
        Price function:
            U = x²/2
            U' = x
            B (beta) = 1
            Y (gamma) = 1/(NB) = 1/N
            p(t) = Y * ( base_load(t) + Σ charging_schedule ) ; n = 1,...,N   t=1,...,T
            p(t) = (1/N)( base_load(t) + Σ charging_schedule )
        """

        # Calculate total charging load of EVs at time t
        ev_load = np.zeros(TIME_HORIZON)
        for t in range(TIME_HORIZON):
            ev_load[t] = 0
            for n in range(N_EV):
                ev_load[t] += charging_schedules[n][t] * MaxPower

        # Calculate total demand at time t (Base load + EV-load)
        total_load = base_load[:TIME_HORIZON] + ev_load

        # Calculate price signal
        price = self.Y * (total_load)

        # Return price signal
        return price

    def calculate_charging_schedule(self, price, worktime_charge_station, t_charge, plug_in_time,
                                plug_out_time, previous_schedule):
        """ Calculate the optimal charging schedule for an EV using Quadratic Optimization.
        Keyword arguments:
            price : Electricity price
            maximum_charging_rate : Maximum allowable charging rate of the EV
            power_req : Total amount of power required by the EV
            plug_in_time : Plug-in time of EV
            plug_out_time : Plug-out time of EV
            previous_schedule : Charging profile of earlier (n)th iteration
        Returns:
            new_schedule : Charging profile of (n+1)th iteration (Updated charging rates during each time slot)
        Optimization:
            At nth iteration,
            Find x(n+1) that,
                Minimize  Σ(Charging cost + penalty term) for t=1,....,number_of_time_slots
                Minimize  Σ {<p(t), (new_schedule)> + 1/2(new_schedule - previous_schedule)²}
        Assumptions:
            All EVs are available for negotiation at the beginning of scheduling period
        """
        # Define variables for new charging rates during each time slot
        # model = gb.Model('UserCost')
        # new_schedule = model.addVars(TIME_HORIZON, vtype=gb.GRB.BINARY)
        # model.update()
        # model.setObjective(price * new_schedule + 0.5 * , sense=gb.GRB.MINIMIZE)

        new_schedule = V(shape=(N_ChargeStation, TIME_HORIZON), integer=True)

        # Define Objective function
        objective = MIN(SM(price @ new_schedule) + 0.5 * SS(new_schedule - previous_schedule))

        # Define constraints list
        constraints = []
        # Constraint for charging rate limits
        constraints.append(0.0 <= new_schedule)
        constraints.append(new_schedule <= 1)
        # Constraint for total amount of power required
        constraints.append(sum(new_schedule) == t_charge)
        # Constraint for specifying EV's arrival & departure times
        if plug_in_time != 0:
            constraints.append(new_schedule[:plug_in_time] == 0)
        if plug_out_time == TIME_HORIZON - 1:
            constraints.append(new_schedule[plug_out_time] == 0)
        elif plug_out_time != TIME_HORIZON:
            constraints.append(new_schedule[plug_out_time:] == 0)

        # Solve the problem
        prob = PB(objective, constraints)
        prob.solve()

        # Solution
        result = (new_schedule.value).tolist()
        return result

    def Solve(self):
        charging_schedules = np.zeros(shape=(N_EV, N_ChargeStation, TIME_HORIZON))
        previous_price = np.zeros(shape=(N_ChargeStation, TIME_HORIZON))
        worktime_charge_station= np.zeros(shape=(N_ChargeStation,N_ChargeUnit,TIME_HORIZON))
        k = 0
        while True:
            # Uncomment:
            print('\nIteration ', k)
            print('----------------------------------------')

            # Step ii
            #   Utility calculates the price control signal and broadcasts to all EVs
            price = self.calculate_price_signal(self.base_load, charging_schedules)

            # Step iii
            # Each EV locally calculates a new charging profile by  solving the optimization problem
            # and reports new charging profile to utility
            new_charging_schedules = np.zeros(shape=(N_EV, TIME_HORIZON))
            # Stop values of all EVs should be true to terminate the algorithm
            stop = np.ones(N_EV, dtype=bool)
            for n in range(N_EV):
                # Uncomment:
                # print('For EV ', n)
                new_charging_schedules[n] = self.calculate_charging_schedule\
                    (price, worktime_charge_station, self.t_charge[n], self.t_plug_in[n], self.t_plug_out[n], charging_schedules[n])

                # Stopping criterion
                # sqrt{(p(k) - p(k-1))²} <= 0.001, for t=1,...,T
                stop[n] = True
                for t in range(TIME_HORIZON):
                    deviation = np.sqrt((price[t] - previous_price[t]) ** 2)
                    if deviation > 0.001:
                        stop[n] = False
                        break

            #evaluation
            aggregate_load = self.base_load
            for t in range(TIME_HORIZON):
                for n in range(N_EV):
                    aggregate_load[t] += charging_schedules[n][t]
            peak = np.amax(aggregate_load)
            print("PEAK: ", peak, 'kW')

            if np.all(stop):
                break
            else:
                # Step iV
                # Go back to Step ii
                charging_schedules = new_charging_schedules
                previous_price = price
                k += 1


        # Remove negative 0 values from output
        charging_schedules[charging_schedules < 0] = 0
        result = np.around(charging_schedules, decimals=2)

def Main():
    np.random.seed(1234)
    generate_ev_data(N_EV)
    model = OffLineModel()
    model.Solve()


if __name__ == "__main__":
    Main()
