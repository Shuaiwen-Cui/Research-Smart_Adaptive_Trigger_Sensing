"""
This script is for sensor on-demand triggering mechanism calculation. The function is able to calculate the triggering mechanism based on the given parameters, and the function is to be called in the main script.

Author: Shuaiwen Cui
Date: May 18, 2024

Log:
-  May 18, 2024: initial version

"""

## activation function
## input: signal, threshold for activation, number of consecutive points to trigger
## output: activation status, index of the point that triggers

def activation(signal, threshold, time):
    # # ensure threshold is positive, and time is positive integer
    # if threshold < 0:
    #     raise ValueError('Threshold must be positive')
    # if time <= 0:
    #     raise ValueError('Time must be positive')
    # time = int(time) # ensure time is an integer
    
    # initialize the activation status
    act_status = 0
    # initialize the trigger index
    trigger_idx = -1
    # initialize the counter
    counter = 0
    
    # squeeze the signal to 1D
    signal = signal.squeeze()
    
    # iterate through the signal
    for i in range(len(signal)):
        # if the signal is above the threshold
        if abs(signal[i]) > threshold:
            # increase the counter
            counter += 1
            # if the counter is above the time
            if counter >= time:
                # set the activation status to 1
                act_status = 1
                # set the trigger index
                trigger_idx = i
                # break the loop
                break
        else:
            # reset the counter
            counter = 0
    # return the activation status and the trigger index
    return act_status, trigger_idx

## inactivation function
## input: signal, threshold for inactivation (must be positive), number of consecutive points to inactivate (must be positive integer)
## output: index of the point that inactivates, if no satisfactory inactivation is found, return the last index of the signal
## no need to return status, because we only call this function when the activation status is 1

def inactivation(signal, threshold, time):
    
    # initialize the counter
    counter = 0
    
    # initialize the inactivation index
    inact_idx = len(signal) - 1
    
    # iterate through the signal
    for i in range(len(signal)):
        # if the signal is below the threshold
        if abs(signal[i]) < threshold:
            # increase the counter
            counter += 1
            # if the counter is above the time
            if counter >= time:
                # set the inactivation index
                inact_idx = i
                # break the loop
                break
        else:
            # reset the counter
            counter = 0
    # return the inactivation index
    return inact_idx