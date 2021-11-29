from data_generate import *
from config import *
import numpy as np
import pandas as pd
from cvxpy import Variable as V, Problem as PB, sum_squares as SS, Minimize as MIN, sum as SM, multiply as MUL, norm as NORM
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
class OffLineModel():
    def __init__(self):
        self.timeline = np.loadtxt('./data/load_data.txt')[:,0]
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
        #plug-expeced time of EVs
        self.t_plug_expected = np.array(self.df_ev['Plug-expected time'].astype(int))
        # Charging efficiency of EVs
        self.efficiency = np.array(self.df_ev['Charging efficiency'])
        #Charging time
        self.t_charge = np.ceil(self.power_req/self.p_max).astype(np.int)
        #location at arrival
        self.loc_at_arrival = np.array(self.df_ev['loc_at_arrival'])
        #destination
        self.loc_at_destination = np.array(self.df_ev['loc_at_destination'])
        #price adjustment constant
        self.Y = Y
        #node distance matrix
        self.dist = np.loadtxt('./data/dist.txt')
        self.timeMatrix = self.dist/v_car
        # Check feasibility of charging
        for n in range(N_EV):
            if (self.t_plug_out[n] - self.t_plug_in[n]) < (self.power_req[n] / self.p_max[n]):
                raise ValueError('Solution is not feasible')

    def calculate_price_signal(self, base_load, ev_load):
        """ Calculate the price signal based on updated charging schedules of EVs.
        Price during time t is modeled as a function of total demand during time t
        """

        # Calculate total demand at time t (Base load + EV-load)
        # total_load = base_load[:TIME_HORIZON] + ev_load
        total_load =ev_load
        # Calculate price signal
        # total_load_eachTime = np.sum(total_load, axis=0)
        load =(price_ub-price_lb) * (total_load-np.min(total_load))/(np.max(total_load)-np.min(total_load)) + price_lb
        load = 1/load

        #define price variable
        new_price = V(shape=(N_ChargeStation, TIME_HORIZON))
        # Define Objective function
        # objective = MIN(SS(MUL(new_price, load)-MUL(new_price, load)/(N_ChargeStation*TIME_HORIZON)))
        objective = 0
        for i in range(TIME_HORIZON):
            objective += ((SM(new_price[:,i] @ load[:,i])-SM(MUL(new_price, load))/(N_ChargeStation*TIME_HORIZON))**2)*weight_time
            objective += SM((new_price[:,i] @ load[:,i] - SM(new_price[:,i] @ load[:,i])/N_ChargeStation)**2)*weight_cs
        objective = MIN(objective)
        # Define constraints list
        constraints = []
        constraints.append(new_price <= price_ub)
        constraints.append(new_price >= price_lb)
        for i in range(N_ChargeStation):
            constraints.append(SM(new_price[i,:]) == TIME_HORIZON * price_mean)

        # Solve the problem
        prob = PB(objective, constraints)
        prob.solve()

        # Solution
        new_price = np.array((new_price.value).tolist())

        # Return price signal
        return new_price

    def calculate_charging_schedule_noneOpt(self, worktime_charge_station, t_charge,plug_in_time, plug_out_time, plug_expected_time, loc_at_arrival):
        worktime_charge_station_min = np.min(worktime_charge_station, axis=1)
        cost_min = np.inf
        for i in range(plug_in_time, plug_out_time-t_charge+1):
            for j in range(N_ChargeStation):
                if worktime_charge_station_min[j, i] > 0:
                    continue
                cost = np.abs(i - plug_expected_time)+self.timeMatrix[loc_at_arrival,j]
                if cost < cost_min and i+t_charge <= plug_out_time and worktime_charge_station_min[j, i+t_charge-1] <= 0.001:
                    cost_min = cost
                    choice_time = i
                    choice_station = j
                    choice_station_unit = np.argmin(worktime_charge_station[choice_station, :, choice_time])

        worktime_charge_station[choice_station, choice_station_unit, choice_time:choice_time+t_charge] = 1
        return choice_time, worktime_charge_station, np.abs(choice_time - plug_expected_time)

    def calculate_charging_schedule(self, price, worktime_charge_station, t_charge, plug_in_time,
                                plug_out_time, previous_schedule, loc_at_arrival, plug_expected_time):
        """ Calculate the optimal charging schedule for an EV using Quadratic Optimization.
        """

        cost_min = np.inf
        worktime_charge_station_min = np.min(worktime_charge_station, axis = 1)
        cost_all = []
        for i in range(plug_in_time, plug_out_time-t_charge+1):
            for j in range(N_ChargeStation):
                if worktime_charge_station_min[j, i] > 0: continue
                cost = sum(price[j, i:i+t_charge])*MaxPower + \
                       self.timeMatrix[loc_at_arrival,j] *Time_Value + \
                       Y*np.abs(i-previous_schedule) + alpha * np.abs(i-plug_expected_time) +\
                       beta * (i < 8 or i > 22)
                if cost < cost_min and i+t_charge <= plug_out_time and worktime_charge_station_min[j, i+t_charge-1] <= 0.001:
                    cost_min = cost
                    choice_time = i
                    choice_station = j
                    choice_station_unit = np.argmin(worktime_charge_station[choice_station, :, choice_time])
            cost_all.append([previous_schedule, cost, sum(price[j, i:i+t_charge])*MaxPower, worktime_charge_station_min[j, i]*Time_Value, self.Y*np.sqrt(np.sum((i-previous_schedule)**2))])
        #update schedual

        worktime_charge_station[choice_station, choice_station_unit, choice_time:choice_time+t_charge] = 1

        return choice_time, worktime_charge_station, np.abs(choice_time-plug_expected_time)

    def ComputeCost(self, ev_load):
        res = 0
        for load in ev_load:
            for i in range(len(ladder_power)):
                res +=  load * ladder_price[i] * (load>ladder_power[i])
        return res

    def ComputeRevenue(self, price, load):
        return np.sum(price*load)

    def ShowResult(self):
        plt.figure()
        plt.subplot(2,1,1)
        #[np.argsort(self.timeline)]
        plt.plot(np.arange(0, TIME_HORIZON, 1), self.aggregate_load, label='Aggregate load')
        plt.plot(np.arange(0, TIME_HORIZON, 1), self.base_load, label='Initial load')
        plt.plot(np.arange(0, TIME_HORIZON, 1), self.aggregate_load_base, label='Aggregate load baseline')
        plt.grid()
        plt.xticks(np.arange(24),
                   ('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9', '10', '11'))
        plt.legend(loc='best')
        plt.title('power load')

        plt.subplot(2, 1, 2)
        plt.plot(np.arange(0, TIME_HORIZON, 1), self.ev_load_base, label='ev load baseline')
        plt.plot(np.arange(0, TIME_HORIZON, 1), self.ev_load, label='ev load')
        plt.grid()
        plt.xticks(np.arange(24),
                   ('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9', '10', '11'))
        plt.legend(loc='best')


        plt.figure()
        for i in range(N_ChargeStation):
            plt.plot(np.arange(0, TIME_HORIZON, 1), self.price[i,:], label ='price:'+str(i))
        plt.xticks(np.arange(24),
                   ('12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '0', '1', '2', '3',
                    '4', '5', '6', '7', '8', '9', '10', '11'))
        plt.legend(loc='best')
        plt.title('price')


        plt.figure()
        for i in range(len(self.iteration_res)):
            plt.plot(self.iteration_res[i][2], label = 'iter:'+str(i))
        plt.legend()
        plt.title('iteration result')

        plt.figure()
        plt.plot(np.arange(len(self.iteration_res)), [self.iteration_res[t][-1] for t in range(len(self.iteration_res))])
        plt.title('var')

        plt.show()

    def SolveNoneOptimal(self):
        charging_schedules = self.t_plug_expected
        worktime_charge_station = np.zeros(shape=(N_ChargeStation, N_ChargeUnit, TIME_HORIZON), dtype=np.int)
        # user wait time
        user_queue_time = np.zeros(shape=(N_EV,))
        for n in range(N_EV):
            # Uncomment:
            # print('For EV ', n)
            _, worktime_charge_station, user_queue_time[n] = self.calculate_charging_schedule_noneOpt \
                (worktime_charge_station, self.t_charge[n], self.t_plug_in[n], self.t_plug_out[n],
                 charging_schedules[n], self.loc_at_arrival[n])
        self.cs_load_base = np.sum(worktime_charge_station > 0, axis=1) * MaxPower
        self.ev_load_base = np.sum(worktime_charge_station > 0, axis=(0, 1)) * MaxPower
        self.aggregate_load_base = self.base_load + self.ev_load_base
        self.peak_base = np.amax(self.aggregate_load_base)
        self.cost_base = self.ComputeCost(self.ev_load_base)



    def Solve(self):
        self.SolveNoneOptimal()

        print("none opt PEAK: ", self.peak_base, 'kW', 'none opt cost:',self.cost_base)
        charging_schedules = self.t_plug_expected  #charge start time
        previous_price = np.zeros(shape=(N_ChargeStation, TIME_HORIZON))
        previous_ev_load = self.ev_load_base
        previous_cs_load = np.zeros(shape=(N_ChargeStation, TIME_HORIZON))+previous_ev_load
        k = 0
        iteration_res = []
        while True:
            # Uncomment:
            print('\nIteration ', k)
            print('----------------------------------------')
            # Step i
            # charge station work time
            worktime_charge_station = np.zeros(shape=(N_ChargeStation, N_ChargeUnit, TIME_HORIZON), dtype = np.int)
            # user wait time
            user_queue_time = np.zeros(shape=(N_EV, ))

            # Step ii
            #   Utility calculates the price control signal and broadcasts to all EVs
            price = self.calculate_price_signal(self.base_load, previous_cs_load)

            # Step iii
            # Each EV locally calculates a new charging profile by  solving the optimization problem
            # and reports new charging profile to utility
            new_charging_schedules = np.zeros(shape=(N_EV, ))
            # Stop values of all EVs should be true to terminate the algorithm
            stop = np.ones(N_EV, dtype=bool)
            for n in range(N_EV):
                # Uncomment:
                # print('For EV ', n)
                new_charging_schedules[n], worktime_charge_station, user_queue_time[n] = self.calculate_charging_schedule\
                    (price, worktime_charge_station, self.t_charge[n], self.t_plug_in[n], self.t_plug_out[n], charging_schedules[n],
                     self.loc_at_arrival[n], self.t_plug_expected[n])

                # Stopping criterion
                # sqrt{(p(k) - p(k-1))²} <= 0.001, for t=1,...,T
                stop[n] = True
                for t in range(TIME_HORIZON):
                    deviation = np.sqrt(np.sum((price[:, t] - previous_price[:, t]) ** 2))
                    if deviation > 0.001:
                        stop[n] = False
                        break

            #evaluation
            previous_cs_load = np.sum(worktime_charge_station > 0, axis=1)*MaxPower
            previous_ev_load = np.sum(worktime_charge_station > 0, axis=(0,1))*MaxPower
            previous_aggregate_load = self.base_load + previous_ev_load
            peak = np.amax(previous_cs_load)
            cost = self.ComputeCost(previous_ev_load)
            revenue = self.ComputeRevenue(price, previous_cs_load)
            iteration_res.append([previous_cs_load, previous_ev_load, previous_aggregate_load,
                                  peak, cost, revenue, user_queue_time, np.var(previous_ev_load)])
            print("PEAK: ", peak, 'kW', 'cost:',cost, 'revenue:', revenue, 'profit:', revenue-cost, 'variance:', np.var(previous_ev_load))


            if np.all(stop) or k > MaxIteration:
                # plt.plot(price[0,:])
                self.ev_load = previous_ev_load
                self.aggregate_load = previous_aggregate_load
                self.price = price
                self.iteration_res = iteration_res
                break
            else:
                # Step iV
                # Go back to Step ii
                charging_schedules = new_charging_schedules
                previous_price = price
                k += 1



def Main():
    np.random.seed(2021)
    generate_ev_data(N_EV)
    model = OffLineModel()
    model.Solve()
    model.ShowResult()


if __name__ == "__main__":
    Main()
