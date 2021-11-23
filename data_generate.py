import scipy.stats as sp
import csv
from config import *

def generate_random_value(mean, standard_deviation):
    """ Generate a random value in a normal distribution.
    Keyword arguments:
        mean : Mean of the distribution
        standard_deviation : Standard deviation of the Gaussian (normal) distribution
    Returns:
        null
    """
    norm1 = sp.norm(loc=mean, scale=standard_deviation)
    generated_number = round(norm1.rvs(), 2)
    # To prevent negative values
    if generated_number < 0:
        generated_number = (-1) * generated_number
    return generated_number


def generate_ev_data(no_of_records):
    """ Generate an EV record (Maximum power,Plug-in time,Plug-out time,SOC at arrival,SOC at departure,Battery capacity,
    Charging efficiency)
    and write to the csv file.
    Keyword arguments:
        no_of_records : Number of EV records to generate
    Returns:
        null
    """

    # Values taken from M. Yilmaz and P. T. Krein, "Review of Battery Charger Topologies, Charging Power Levels, and " \
    # "Infrastructure for Plug-In Electric and Hybrid Vehicles," in IEEE Transactions on Power Electronics, vol. 28,
    # no. 5, pp. 2151-2169, May 2013
    # Toyota Prius PHEV: (3.8kW, 4.4kWh)
    # Chevrolet PHEV: (3.8kW, 16kWh)
    # Mitsubishi i-MiEV EV: (3kW, 16kWh)
    # Nissan Leaf EV: (3.3kW, 24kWh)

    # EV Id
    ev_id = [i for i in range(1, no_of_records + 1)]

    # 2) Maximum charging rate of EVs
    maximum_power = [MaxPower*24/TIME_HORIZON] * no_of_records
    # for n in range(no_of_records):
    #     maximum_power.append(random.choice([3.8, 3.8, 3, 3.3]))

    # Plug-in times and Plug-out times of EVs are assumed to be Gaussian.
    # It is assumed that EVs leave home at 7 am (7 -> 19) with standard deviation of 1h and
    #   come back home at 5 pm (17 -> 5) with standard deviation of 2h
    # 3) Plug-in times of EVs
    plug_in_times = []
    for i in range(no_of_records):
        plug_in_times.append(int(round(generate_random_value(1, 1))))
    # 4) Plug-out times of EVs
    plug_out_times = []
    for i in range(no_of_records):
        plug_out_times.append(min(int(round(generate_random_value(23, 1))), TIME_HORIZON))

    # Required SoC at plug-out time is assumed be 90% of the total capacity to avoid premature aging
    # 6) SoC of EVs departure
    soc_at_departure = [1] * no_of_records

    # SoC at the plug-in time is assumed Gaussian with a mean of o.3 and a standard deviation of 0.1
    # 5) SoC of EVs at arrival
    soc_at_arrival = []
    for i in range(no_of_records):
        # To ensure that soc at arrival is always less than soc at departure
        while True:
            soc_in = generate_random_value(0.3, 0.1)
            if soc_in < soc_at_departure[i]:
                break
        soc_at_arrival.append(soc_in)

    # 7) Battery capacities of EVs
    capacities = [MaxCapacities] * no_of_records
    # capacities = []
    # for i in maximum_power:
    #     if i == 3.8:
    #         capacities.append(random.choice([4.4, 16]))
    #     elif i == 3:
    #         capacities.append(16)
    #     elif i == 3.3:
    #         capacities.append(24)

    # 8) Charging efficiency is assumed 85%
    efficiencies = [Efficiencies] * no_of_records

    # Write to data/ev.csv
    # --------------------

    # Open the file for writing
    csv_out = open('./data/ev.csv', 'w')
    # Create the csv writer object
    mywriter = csv.writer(csv_out)
    # Write the header
    mywriter.writerow(["Maximum power", "Plug-in time", "Plug-out time", "SOC at arrival", "SOC at departure",
                       "Battery capacity", "Charging efficiency"])
    # Write all rows at once
    rows = zip(ev_id, maximum_power, plug_in_times, plug_out_times, soc_at_arrival, soc_at_departure, capacities,
               efficiencies)
    mywriter.writerows(rows)
    # Close the file
    csv_out.close()