import numpy as np
import os
import datetime


def find_sub(string, sub):
    '''finds all instances of a substring within a string and outputs a list of indices'''
    result = []
    k = 0
    while k < len(string):
        k = string.find(sub, k)
        if k == -1:
            return result
        else:
            result.append(k)
            k += 1  # change to k += len(sub) to not search overlapping results
    return result


def MatlabNumSeq(start, stop, step, exclude=True):
    """In Matlab you can type:

    start:step:stop and easily create a numerical sequence

    if exclude is true it will exclude any values greater than the stop value
    """

    '''np.arange(start, stop, step) works good most of the time

    However, if the step (stop-start)/step is an integer, then the sequence
    will stop early'''

    seq = np.arange(start, stop + step, step)

    if exclude:
        print('hi')
        if seq[-1] > stop:
            seq = seq[:-1]

    return seq


def find_bin_basenames(directory):
    file_list = os.listdir(directory)

    tint_basenames = [os.path.splitext(file)[0] for file in file_list if '.bin' in file]

    return tint_basenames


def find_converted_bin_basenames(directory):
    file_list = os.listdir(directory)

    mda_basenames = [os.path.splitext(file)[0] for file in file_list if '_filt.mda' in file]
    mda_basenames = [file[:find_sub(file, '_')[-1]] for file in mda_basenames]

    tint_basenames = []

    for file in mda_basenames:
        basename = file[:find_sub(file, '_')[-1]]
        if basename not in tint_basenames:
            tint_basenames.append(basename)

    return tint_basenames


def session_datetime(file, output='datetime'):
    '''Getting the Trial Date and Time value for the .set file'''

    file = os.path.splitext(os.path.basename(file))[0]
    date, time = (file[find_sub(file, '_')[-2] + 1:]).split('_')

    date = datetime.datetime.strptime(str(date), '%y%m%d')
    time = datetime.datetime.strptime(str(time), '%H%M%S')
    if output == 'datetime':
        date = date.strftime("%A, %d %b %Y")
        time = time.strftime("%H:%M:%S")

        return date, time

    elif output == 'seconds':
        # returns seconds since epoc
        date = str(date)
        date = date[:date.find(' ')]
        time = str(time)
        time = time[time.find(' ') + 1:]

        seconds = (datetime.datetime.strptime('%s %s' % (str(date), str(time)),
                                              '%Y-%m-%d %H:%M:%S') - datetime.datetime(1970, 1, 1)).total_seconds()
        return seconds