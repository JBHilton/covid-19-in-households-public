import numpy as np

threshold = 0.5

t0 = 0
x0 = 0

tmax = 100

x_old = x0
x_sum = 0

def compare_against_threshold(r, x_old):
    if r > threshold:
        return(x_old + 1)
    else:
        return(x_old - 1)

for t in range(t0, tmax):
    r = np.random.uniform()
    x_new = compare_against_threshold(r, x_old)
    x_sum = x_new
    x_old = x_new

print(x_sum)

def compare_against_threshold(r, x_old, threshold):
    if r > threshold:
        return(x_old + 1)
    else:
        return(x_old - 1)

def generate_rw_sum(threshold):
    t0 = 0
    x0 = 0

    tmax = 100

    x_old = x0
    x_sum = 0
    for t in range(t0, tmax):
        r = np.random.uniform()
        x_new = compare_against_threshold(r, x_old, threshold)
        x_sum = x_new
        x_old = x_new
    return(x_sum)

samples = [generate_rw_sum(threshold) for threshold in np.arange(.1, .9, .1)]