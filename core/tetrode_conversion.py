import numpy as np
from core.readMDA import readMDA
import os
from core.utils import find_sub
from core.Tint_Matlab import get_setfile_parameter
from core.cut_creation import create_cut
from core.rhd_utils import intan_scalar
import struct
import datetime
import json


def get_tetrode_parameters(set_filename):
    parameters = ['trial_date', 'trial_time', 'experimenter', 'comments', 'duration', 'sw_version']

    # tetrode_parameters = {'samples_per_spike': samples_per_spike,
    #                      'rawRate': str(int(rawRate))}

    tetrode_parameters = {}

    for x in parameters:
        tetrode_parameters[x] = get_setfile_parameter(x, set_filename)

    return tetrode_parameters


def write_tetrode_header(filepath, tetrode_parameters):

    date = 'trial_date %s' % (tetrode_parameters['trial_date'])
    time_head = '\ntrial_time %s' % (tetrode_parameters['trial_time'])
    experimenter = '\nexperimenter %s' % (tetrode_parameters['experimenter'])
    comments = '\ncomments %s' % (tetrode_parameters['comments'])

    duration = '\nduration %s' % (tetrode_parameters['duration'])

    sw_vers = '\nsw_version %s' % (tetrode_parameters['sw_version'])

    write_order = [date, time_head, experimenter, comments, duration, sw_vers,
                   # num_chans, timebase_head,
                   # bp_timestamp,
                   # samps_per_spike, sample_rate, b_p_sample, spike_form,
                   # num_spikes, start,
                   # content
                   ]

    with open(filepath, 'rb+') as f:
        # get the already written tetrode data
        content = f.read()

    header_written = 0
    for value in write_order:
        if bytes(value, 'utf-8') in content:
            header_written += 1

    if len(write_order) == header_written:
        # header values already written
        return

    with open(filepath, 'w') as f:
        f.writelines(write_order)

    with open(filepath, 'rb+') as f:
        f.seek(0, 2)
        f.write(content)


def write_tetrode_data(filepath, spike_times, spike_data, samples_per_spike=50, Fs=48000):
    with open(filepath, 'w') as f:
        num_chans = '\nnum_chans 4'
        timebase_head = '\ntimebase %d hz' % (96000)
        bp_timestamp = '\nbytes_per_timestamp %d' % (4)
        # samps_per_spike = '\nsamples_per_spike %d' % (int(Fs*1e-3))
        samps_per_spike = '\nsamples_per_spike %d' % (int(samples_per_spike))
        sample_rate = '\nsample_rate %d hz' % (int(Fs))
        b_p_sample = '\nbytes_per_sample %d' % (1)
        # b_p_sample = '\nbytes_per_sample %d' % (4)
        spike_form = '\nspike_format t,ch1,t,ch2,t,ch3,t,ch4'

        num_spikes = '\nnum_spikes %d' % (spike_data.shape[1])
        start = '\ndata_start'

        write_order = [num_chans, timebase_head,
                       bp_timestamp,
                       samps_per_spike, sample_rate, b_p_sample, spike_form,
                       num_spikes, start]

        f.writelines(write_order)

    spike_data = np.swapaxes(spike_data, 0, 1)

    n, n_channels, clip_size = spike_data.shape  # n spikes

    # re-shaping the data so that the channels are concatenated such that
    # the 0th dimension is n_channels * n_spikes, and the 1st dimension of the array
    # is the clip size (samples per spike).
    spike_data = spike_data.reshape((n_channels * n, clip_size))

    # when writing the spike times we write it for each channel, so lets tile
    spike_times = np.tile(spike_times, (n_channels, 1))
    spike_times = spike_times.flatten(order='F')

    # this will create a (n_samples, n_channels, n_samples_per_spike) => (n, 4, 50) sized matrix, we will create a
    # matrix of all the samples and channels going from ch1 -> ch4 for each spike time
    # time1 ch1_data
    # time1 ch2_data
    # time1 ch3_data
    # time1 ch4_data
    # time2 ch1_data
    # time2 ch2_data
    # .
    # .
    # .
    spike_array = np.hstack((spike_times.reshape(len(spike_times), 1), spike_data))

    spike_times = None
    spike_values = None

    spike_n = spike_array.shape[0]
    t_packed = struct.pack('>%di' % spike_n, *spike_array[:, 0].astype(int))
    spike_array = spike_array[:, 1:]  # removing time data from this matrix to save memory

    spike_data_pack = struct.pack('<%db' % (spike_n * clip_size), *spike_array.astype(int).flatten())

    spike_array = None

    # now we need to combine the lists by alternating

    comb_list = [None] * (2 * spike_n)
    comb_list[::2] = [t_packed[i:i + 4] for i in range(0, len(t_packed), 4)]  # breaks up t_packed into a list,
    # each timestamp is one 4 byte integer
    comb_list[1::2] = [spike_data_pack[i:i + 50] for i in range(0, len(spike_data_pack), 50)]  # breaks up spike_data_
    # pack and puts it into a list, each spike is 50 one byte integers

    t_packed = None
    spike_data_pack = None

    write_order = []
    with open(filepath, 'rb+') as f:
        write_order.extend(comb_list)
        write_order.append(bytes('\r\ndata_end\r\n', 'utf-8'))

        f.seek(0, 2)
        f.writelines(write_order)


def get_clip_values(data_masked, spike_times, spike_channel, remove_spike_percentage=1,
                    detect_sign=0, remove_outliers=False, method='max', clip_scalar=1):
    """

    method = 'max'  # it will find the max (or min in case of negative peaks)

    """

    # ----------- calculate the peaks for each cell -------------
    # peak_values = np.sort(data_masked[:, spike_times].flatten())
    peak_values = np.array([])
    for chan in np.unique(spike_channel):
        chan_bool = np.where(spike_channel == chan)[0]
        chan_spikes = data_masked[chan - 1, spike_times[chan_bool]]

        if detect_sign == 1:
            # just make sure that no spike times with negative peaks leak in
            sign_bool = np.where(chan_spikes >= 0)[0]
            chan_spikes = chan_spikes[sign_bool]
        elif detect_sign == -1:
            # just make sure no spike times with positive peaks leak in
            sign_bool = np.where(chan_spikes <= 0)[0]
            chan_spikes = chan_spikes[sign_bool]

        peak_values = np.hstack((peak_values, chan_spikes))

    peak_values = np.sort(peak_values.flatten())

    n_spikes = len(peak_values)
    remove_n = int(n_spikes * (remove_spike_percentage / 100))  # how many

    if remove_outliers:
        # hypothetically if you have high amplitude noise that is significantly larger from the
        # putative spikes, if you force the highest peak to be the max bit value it will shrink
        # the appearance of the putative spikes. So if you want to remove outliers this could benefit
        # you by not washing out the smaller amplitude peaks. Hopefully the data masking removes those
        # high amplitude spikes though.
        if detect_sign == 0:
            # there are positive and negative peaks to remove
            # split the number of spikes to remove evenly between positive and negative peaks
            remove_n = int(remove_n / 2)
            peak_values = peak_values[remove_n:-remove_n]
        elif detect_sign == 1:
            # positive peaks, remove some of the positive peak values
            peak_values = peak_values[:-remove_n]
        elif detect_sign == -1:
            # negative peaks, remove some of the negative peak values
            peak_values = peak_values[remove_n:]

    if method == 'max':
        clip_value = np.amax(np.abs([np.amax(peak_values), np.amin(peak_values)]))
    elif method == 'median':
        clip_value = np.abs(np.median(peak_values))
    elif method == 'mean':
        clip_value = np.abs(np.mean(peak_values))
    else:
        raise ValueError("error in the clipping method! Should be 'max', 'median', or 'mean'.")

    return clip_scalar * clip_value


def convert_tetrode(filt_filename, output_basename, Fs, pre_spike_samples=10, post_spike_samples=40, detect_sign=0,
                    remove_spike_percentage=1,  remove_outliers=False, clip_method='max', clip_scalar=1, self=None):
    """
    convert_tetrode will take an input filename that was processed and sorted via MountainSort
    and convert the MountainSorted output to a Tint output.

    Args:
        filt_filename (str): this is the fullpath of the file that was pre-processed/sorted using MountainSort
        self (bool): this refers to the self of a QtGui, that has a QTextEdit displaying logs. There is a method called
            LogAppend that prints the current status of the Gui to the TextEdit.
    Returns:
        None
    """

    msg = '[%s %s]: Converting the MountainSort output following filename to Tint: %s' % (str(datetime.datetime.now().date()),
                                                                            str(datetime.datetime.now().time())[
                                                                            :8], filt_filename)

    tetrode_skipped = False

    if self is None:
        print(msg)
    else:
        self.LogAppend.myGUI_signal_str.emit(msg)

    directory = os.path.dirname(filt_filename)

    mda_basename = os.path.splitext(filt_filename)[0]
    mda_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

    tint_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

    # set_filename = '%s.set' % (os.path.join(directory, tint_basename))
    # set_filename = '%s.set' % output_basename

    cell_numbers = None

    tetrode = int(mda_basename[find_sub(mda_basename, '_')[-1] + 2:])

    # new_basename = '%s_ms' % tint_basename
    # tetrode_filepath = '%s.%d' % (os.path.join(directory, new_basename), tetrode)

    tetrode_filepath = '%s.%d' % (output_basename, tetrode)

    # get the tint spike information, placing the event at the 11th index
    pre_spike = pre_spike_samples + 1
    post_spike = post_spike_samples - 1

    if os.path.exists(tetrode_filepath):
        msg = '[%s %s]: The following file already exists: %s, skipping conversion!' % (
        str(datetime.datetime.now().date()),
        str(datetime.datetime.now().time())[
        :8], tetrode_filepath)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)

        tetrode_skipped = True

    else:
        masked_out_fname = mda_basename + '_masked.mda'
        firings_out = mda_basename + '_firings.mda'

        # ----------- reading in mountainsort spike data ------------------------- #

        if not os.path.exists(firings_out):

            msg = '[%s %s]: The following spike filename does not exist: %s, skipping!' % (
                str(datetime.datetime.now().date()),
                str(datetime.datetime.now().time())[
                :8], firings_out)

            if self is None:
                print(msg)
            else:
                self.LogAppend.myGUI_signal_str.emit(msg)

            raise FileNotFoundError('Could not find the following spike filename: %s' % firings_out)

        msg = '[%s %s]: Reading the spike data from the following file: %s' % (
            str(datetime.datetime.now().date()),
            str(datetime.datetime.now().time())[
            :8], firings_out)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)

        A, _ = readMDA(firings_out)

        spike_channel = A[0, :].astype(int)
        spike_times = A[1, :].astype(int)  # at this stage it is in index values (0-based)
        cell_numbers = A[2, :].astype(int)
        # ------------- creating clips ---------------------- #

        if not os.path.exists(masked_out_fname):

            msg = '[%s %s]: The following masked data filename does not exist: %s, skipping!' % (
                str(datetime.datetime.now().date()),
                str(datetime.datetime.now().time())[
                :8], masked_out_fname)

            if self is None:
                print(msg)
            else:
                self.LogAppend.myGUI_signal_str.emit(msg)

            raise FileNotFoundError('Could not find the following masked data: %s' % masked_out_fname)

        msg = '[%s %s]: Reading the spike data from the following file: %s' % (
            str(datetime.datetime.now().date()),
            str(datetime.datetime.now().time())[
            :8], masked_out_fname)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)

        # read in masked data
        data_masked, _ = readMDA(masked_out_fname)

        clip_size = pre_spike + post_spike

        # max sample index
        max_n = data_masked.shape[1] - 1

        spike_bool = np.where((spike_times + post_spike < max_n) * (spike_times - pre_spike >= 0))[0]

        spike_channel = spike_channel[spike_bool]
        spike_times = spike_times[spike_bool]
        cell_numbers = cell_numbers[spike_bool]

        spike_bool = None

        clip_indices = np.tile(spike_times, (clip_size, 1)).T + np.arange(-pre_spike, post_spike)

        # getting the clip values

        # ------------------
        # converting the tetrode data from 16bit at 192gain for 2.45 v-swing to 1500mV half v-swing

        # convert data to uV
        data_masked = data_masked * intan_scalar()

        # the value representing the half bit range in uV (128 bit value represents this in uV)
        channel_clip = get_clip_values(data_masked, spike_times, spike_channel, detect_sign=detect_sign,
                                       remove_spike_percentage=remove_spike_percentage,
                                       method=clip_method, clip_scalar=clip_scalar,
                                       remove_outliers=remove_outliers)

        ADC_Fullscale = 1500  # mV, this is what the half voltage swing is in Tint

        # calculates the gain so that the full-scale is 1500mV
        # (A value of 128 bits with a gain of 1, would represent a 1500mV value)
        # round up to include the channel clip, tint likes round channel gains
        channel_gains = int(np.ceil(ADC_Fullscale / (channel_clip / 1000)))

        # the new channel clip value with the rounded gain, bits
        channel_clip = 128 * (ADC_Fullscale * 1000 / (channel_gains * 128))

        # calculate a scalar that will be used to convert data from uV to bits (8bit data)
        channel_scalar8s = channel_clip / 128

        # calculates the scalar value to divide the uV value to convert to 16 bit
        channel_scalar16s = channel_clip / 32768

        # ------------------- getting spike times --------------------------------- #

        cell_data = data_masked[:, clip_indices]

        # converting data to 8 bit with a max half scale of 1500mV
        cell_data = (cell_data / channel_scalar8s).astype(int)
        # cell_data = np.divide(cell_data, 256).astype(int)  # converting from int16 back to int8

        # ensuring that the data is within the right integer range
        cell_data[np.where(cell_data > 127)] = 127
        cell_data[np.where(cell_data < -128)] = -128

        tetrode_clip = {'gain': channel_gains, 'clip_value': channel_clip, 'scalar8bit': channel_scalar8s,
                        'scalar16bit': channel_scalar16s}

        clip_filename = '%s_clips.json' % tint_basename

        # save the clip and gain data to be saved later in the set file
        if os.path.exists(clip_filename):
            with open(clip_filename, 'r') as f:
                clip_data = json.load(f)

            clip_data[tetrode] = tetrode_clip

            with open(clip_filename, 'w') as f:
                json.dump(clip_data, f)
        else:
            with open(clip_filename, 'w') as f:
                json.dump({tetrode: tetrode_clip}, f)

        data_masked = None

        # Fs = int(get_setfile_parameter('rawRate', set_filename))

        cell_times = (spike_times * (96000 / Fs)).astype(
            int)  # need to convert to the 96000 Hz timebase that Tint has, time occurs at the 12th value

        # tetrode_parameters = get_tetrode_parameters(set_filename, samples_per_spike=clip_size, rawRate=Fs)

        msg = '[%s %s]: Creating the following tetrode file: %s!' % (
            str(datetime.datetime.now().date()),
            str(datetime.datetime.now().time())[
            :8], tetrode_filepath)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)

        write_tetrode_data(tetrode_filepath, cell_times, cell_data, samples_per_spike=clip_size, Fs=Fs)

    # ------------ creating the cut file ----------------------- #

    # output_basename = '%s_ms' % tint_basename
    clu_filename = '%s.clu.%d' % (os.path.join(directory, output_basename), tetrode)
    cut_filename = '%s_%d.cut' % (os.path.join(directory, output_basename), tetrode)

    if os.path.exists(cut_filename):
        msg = '[%s %s]: The following cut file already exists: %s, skipping conversion!' % (
            str(datetime.datetime.now().date()),
            str(datetime.datetime.now().time())[
            :8], cut_filename)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)
    else:

        msg = '[%s %s]: Creating the following cut file: %s!' % (
            str(datetime.datetime.now().date()),
            str(datetime.datetime.now().time())[
            :8], cut_filename)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)

        if tetrode_skipped:
            firings_out = mda_basename + '_firings.mda'
            masked_out_fname = mda_basename + '_masked.mda'

            A, _ = readMDA(firings_out)

            spike_times = (A[1, :]).astype(int)  # at this stage it is in index values (0-based)
            cell_numbers = A[2, :].astype(int)

            data_masked, _ = readMDA(masked_out_fname)

            # max sample index
            max_n = data_masked.shape[1] - 1

            # making sure now
            spike_bool = np.where((spike_times + post_spike < max_n) * (spike_times - pre_spike >= 0))[0]

            spike_times = None
            cell_numbers = cell_numbers[spike_bool]

        create_cut(cut_filename, clu_filename, cell_numbers, tetrode, tint_basename, output_basename, self=self)


def batch_basename_tetrodes(directory, tint_basename, output_basename, Fs, pre_spike_samples=10, post_spike_samples=40,
                            detect_sign=0, remove_spike_percentage=1, remove_outliers=False, clip_scalar=1,
                            clip_method='max', self=None):

    # find the filenames that were used by MountainSort to be sorted.
    filt_fnames = [os.path.join(directory, file) for file in os.listdir(
        directory) if '_filt.mda' in file if os.path.basename(tint_basename) in file]

    for file in filt_fnames:
        try:
            convert_tetrode(file, output_basename, Fs, pre_spike_samples=pre_spike_samples,
                            post_spike_samples=post_spike_samples, detect_sign=detect_sign,
                            remove_spike_percentage=remove_spike_percentage,  remove_outliers=remove_outliers,
                            clip_method=clip_method,
                            clip_scalar=clip_scalar,
                            self=self)

        except FileNotFoundError:
            continue


def is_tetrode(file, session):

    if os.path.splitext(file)[0] == session:
        try:
            tetrode_number = int(os.path.splitext(file)[1][1:])
            return True
        except ValueError:
            return False
    else:

        return False


def batch_add_tetrode_headers(directory, tint_fullpath, self=None):
    output_basename = os.path.basename(tint_fullpath) + '_ms'

    # find the filenames that were used by MountainSort to be sorted.
    tetrode_filenames = [os.path.join(directory, file) for file in os.listdir(
        directory) if output_basename in file if is_tetrode(file, output_basename)]

    set_filename = '%s.set' % os.path.join(directory, output_basename)

    # tetrode_parameters = get_tetrode_parameters(set_filename, samples_per_spike=clip_size, rawRate=Fs)
    tetrode_parameters = get_tetrode_parameters(set_filename)

    for file in tetrode_filenames:
        try:
            write_tetrode_header(file, tetrode_parameters)
        except FileNotFoundError:
            continue
