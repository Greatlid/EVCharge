from data_generate import *
from config import *
import numpy as np
import pandas as pd
from cvxpy import Variable as V, Problem as PB, sum_squares as SS, Minimize as MIN, sum as SM, multiply as MUL, norm as NORM
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
        self.t_charge = np.ceil(self.power_req/self.p_max).astype(np.int)
        #price adjustment constant
        self.Y = random.uniform(0, (1 / N_EV))
        # Check feasibility of charging
        for n in range(N_EV):
            if (self.t_plug_out[n] - self.t_plug_in[n]) < (self.power_req[n] / self.p_max[n]):
                raise ValueError('Solution is not feasible')

    def calculate_price_signal(self, base_load, worktime):
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
        ev_load = np.zeros(shape=(N_ChargeStation, TIME_HORIZON))

        for t in range(TIME_HORIZON):
            for n in range(N_ChargeStation):
                ev_load[n][t] += np.sum(worktime[n,:,t] > 0) * MaxPower

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
        new_schedule = np.zeros_like(previous_schedule)
        cost_min = np.inf
        queue_time = np.min(worktime_charge_station, axis = 1)
        for i in range(plug_in_time, plug_out_time-t_charge+1):
            for j in range(N_ChargeStation):
                new_schedule[j, i:i + t_charge] = 1
                cost = sum(price[j, i:i+t_charge]) + queue_time[j, i]*Time_Value \
                       + 0.5*np.sqrt(np.sum((new_schedule-previous_schedule)**2))
                if cost < cost_min and queue_time[j, i]+t_charge < plug_out_time:
                    choice_time = i
                    choice_station = j
                    choice_station_unit = np.argmin(worktime_charge_station[choice_station, :, choice_time])
        #update schedual
        new_schedule[choice_station, choice_time:choice_time+t_charge] = 1
        #update wait time
        wait_time_previous = worktime_charge_station[choice_station, choice_station_unit, choice_time]
        worktime_charge_station[choice_station, choice_station_unit, choice_time] += t_charge
        wait_time_new = worktime_charge_station[choice_station, choice_station_unit, choice_time]
        worktime_charge_station[choice_station, choice_station_unit, choice_time:choice_time+wait_time_new] = \
            np.arange(wait_time_new, max(choice_time+wait_time_new-TIME_HORIZON, 0), step = -1)
        return new_schedule, worktime_charge_station, wait_time_previous

    def ShowResult(self):
        plt.plot(np.arange(0, TIME_HORIZON, 1), self.aggregate_load, label='Aggregate load')
        plt.plot(np.arange(0, TIME_HORIZON, 1), self.base_load, label='Initial load')
        plt.grid()
        plt.legend(loc='best')
        plt.show()

    def Solve(self):
        charging_schedules = np.zeros(shape=(N_EV, N_ChargeStation, TIME_HORIZON))  #charge start time
        previous_price = np.zeros(shape=(N_ChargeStation, TIME_HORIZON))

        k = 0
        while True:
            # Uncomment:
            print('\nIteration ', k)
            print('----------------------------------------')
            # charge station work time
            worktime_charge_station = np.zeros(shape=(N_ChargeStation, N_ChargeUnit, TIME_HORIZON), dtype = np.int)
            # user wait time
            user_queue_time = np.zeros(shape=(N_EV, ))

            # Step ii
            #   Utility calculates the price control signal and broadcasts to all EVs
            price = self.calculate_price_signal(self.base_load, worktime_charge_station)

            # Step iii
            # Each EV locally calculates a new charging profile by  solving the optimization problem
            # and reports new charging profile to utility
            new_charging_schedules = np.zeros(shape=(N_EV, N_ChargeStation, TIME_HORIZON))
            # Stop values of all EVs should be true to terminate the algorithm
            stop = np.ones(N_EV, dtype=bool)
            for n in range(N_EV):
                # Uncomment:
                # print('For EV ', n)
                new_charging_schedules[n], worktime_charge_station, user_queue_time[n] = self.calculate_charging_schedule\
                    (price, worktime_charge_station, self.t_charge[n], self.t_plug_in[n], self.t_plug_out[n], charging_schedules[n])

                # Stopping criterion
                # sqrt{(p(k) - p(k-1))²} <= 0.001, for t=1,...,T
                stop[n] = True
                for t in range(TIME_HORIZON):
                    deviation = np.sqrt(np.sum((price[:, t] - previous_price[:, t]) ** 2))
                    if deviation > 0.001:
                        stop[n] = False
                        break

            #evaluation
            aggregate_load = self.base_load + np.sum(worktime_charge_station > 0, axis=(0,1))*MaxPower
            peak = np.amax(aggregate_load)
            print("PEAK: ", peak, 'kW')

            if np.all(stop):
                self.aggregate_load = aggregate_load
                break
            else:
                # Step iV
                # Go back to Step ii
                charging_schedules = new_charging_schedules
                previous_price = price
                k += 1


        # Remove negative 0 values from output
        charging_schedules[charging_schedules < 0] = 0


def Main():
    np.random.seed(1234)
    generate_ev_data(N_EV)
    model = OffLineModel()
    model.Solve()
    model.ShowResult()


if __name__ == "__main__":
    Main()
