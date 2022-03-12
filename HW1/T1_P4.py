#####################
# CS 181, Spring 2022
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'data/year-sunspots-republicans.csv'
years = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_counts.append(float(row[2]))

# Turn the data into numpy arrays.
years = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985



# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (d) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
# is_years is a Boolean variable which indicates whether or not the input variable is
# years; if so, is_years should be True, and if the input variable is sunspots, is_years
# should be false
def make_basis(xx, part='a', is_years=True):
    # DO NOT CHANGE LINES 65-69
    if part == 'a' and is_years:
        xx = (xx - np.array([1960]*len(xx)))/40
        
    if part == "a" and not is_years:
        xx = xx/20

    basis = None
    len_x = len(xx)
    if part == "a":
        basis = np.ones((len_x, 6))
        for d in range(1, 6):
            for i in range(len_x):
                basis[i, d] = xx[i] ** d
    if part == "b":
        basis = np.ones((len_x, 12))
        for d in range(1, 12):
            for i in range(len_x):
                basis[i, d] = np.exp(-((xx[i]-(5*d+1955)) ** 2) / 25)
    if part == "c":
        basis = np.ones((len_x, 6))
        for d in range(1, 6):
            for i in range(len_x):
                basis[i, d] = np.cos(xx[i]/d)
    if part == "d":
        basis = np.ones((len_x, 26))
        for d in range(1, 26):
            for i in range(len_x):
                basis[i, d] = np.cos(xx[i]/d)

    return basis

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X, Y):
    w = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    return w


if __name__ == "__main__":
    # Plot the data.
    # plt.figure(1)
    # plt.plot(years, republican_counts, 'o')
    # plt.xlabel("Year")
    # plt.ylabel("Number of Republicans in Congress")
    # plt.title("Number of Republicans in Congress by year")
    # plt.figure(2)
    # plt.plot(years, sunspot_counts, 'o')
    # plt.xlabel("Year")
    # plt.ylabel("Number of Sunspots")
    # plt.title("Number of Sunspots by year")
    # plt.figure(3)
    # plt.plot(sunspot_counts[years < last_year], republican_counts[years < last_year], 'o')
    # plt.xlabel("Number of Sunspots")
    # plt.ylabel("Number of Republicans in Congress")
    # plt.title("Number of Republicans in Congress vs number of sunspots")
    # plt.show()

    # Create the simplest basis, with just the time and an offset.
    X = np.vstack((np.ones(years.shape), years)).T

    # Nothing fancy for outputs.
    Y = republican_counts

    w = find_weights(X, Y)

    # Compute the regression line on a grid of inputs.
    # DO NOT CHANGE grid_years!!!!!
    grid_years = np.linspace(1960, 2005, 200)
    grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
    grid_Yhat = np.dot(grid_X.T, w)

    # Problem 4.1: plot and report sum of squared error for each basis

    # Plot the data and the regression line.
    # plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
    # plt.xlabel("Year")
    # plt.ylabel("Number of Republicans in Congress")
    # plt.show()

    basis_a = make_basis(years, is_years=True, part="a")
    basis_b = make_basis(years, is_years=True, part="b")
    basis_c = make_basis(years, is_years=True, part="c")
    basis_d = make_basis(years, is_years=True, part="d")

    w_a = find_weights(basis_a, Y)
    w_b = find_weights(basis_b, Y)
    w_c = find_weights(basis_c, Y)
    w_d = find_weights(basis_d, Y)

    yhat_a = np.dot(basis_a, w_a)
    yhat_b = np.dot(basis_b, w_b)
    yhat_c = np.dot(basis_c, w_c)
    yhat_d = np.dot(basis_d, w_d)

    plt.scatter(years, Y)
    plt.plot(years, yhat_a)
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Republicans by year (basis a)")
    plt.show()

    plt.scatter(years, Y)
    plt.plot(years, yhat_b)
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Republicans by year (basis b)")
    plt.show()

    plt.scatter(years, Y)
    plt.plot(years, yhat_c)
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Republicans by year (basis c)")
    plt.show()

    plt.scatter(years, Y)
    plt.plot(years, yhat_d)
    plt.xlabel("Year")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Republicans by year (basis d)")
    plt.show()

    error_41a = np.sum((Y - yhat_a) ** 2)
    error_41b = np.sum((Y - yhat_b) ** 2)
    error_41c = np.sum((Y - yhat_c) ** 2)
    error_41d = np.sum((Y - yhat_d) ** 2)

    # Problem 4.2

    sunspots_1985 = sunspot_counts[years < 1985]
    Y_1985 = Y[years < 1985]

    basis_a = make_basis(sunspots_1985, is_years=False, part="a")
    basis_c = make_basis(sunspots_1985, is_years=False, part="c")
    basis_d = make_basis(sunspots_1985, is_years=False, part="d")
    w_a = find_weights(basis_a, Y_1985)
    w_c = find_weights(basis_c, Y_1985)
    w_d = find_weights(basis_d, Y_1985)
    yhat_a = np.dot(basis_a, w_a)
    yhat_c = np.dot(basis_c, w_c)
    yhat_d = np.dot(basis_d, w_d)

    sunspots_1985_sorted = sorted(sunspots_1985)
    yhat_a_sorted = [_[1] for _ in sorted(zip(sunspots_1985, yhat_a))]
    yhat_c_sorted = [_[1] for _ in sorted(zip(sunspots_1985, yhat_c))]
    yhat_d_sorted = [_[1] for _ in sorted(zip(sunspots_1985, yhat_d))]

    plt.scatter(sunspots_1985, Y_1985)
    plt.plot(sunspots_1985_sorted, yhat_a_sorted)
    plt.xlabel("Number of sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Republicans vs number of sunspots (basis a)")
    plt.show()

    plt.scatter(sunspots_1985, Y_1985)
    plt.plot(sunspots_1985_sorted, yhat_c_sorted)
    plt.xlabel("Number of sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Republicans vs number of sunspots (basis c)")
    plt.show()

    plt.scatter(sunspots_1985, Y_1985)
    plt.plot(sunspots_1985_sorted, yhat_d_sorted)
    plt.xlabel("Number of sunspots")
    plt.ylabel("Number of Republicans in Congress")
    plt.title("Predicted Republicans vs number of sunspots (basis d)")
    plt.show()

    error_42a = np.sum((Y_1985 - yhat_a) ** 2)
    error_42c = np.sum((Y_1985 - yhat_c) ** 2)
    error_42d = np.sum((Y_1985 - yhat_d) ** 2)
    breakpoint()
