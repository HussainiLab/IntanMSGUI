import os
from core.Tint_Matlab import int16toint8, get_setfile_parameter
import core.filtering as filt
from core.utils import find_sub
from core.readMDA import readMDA
from core.rhd_utils import tetrode_map, intan_scalar
from core.intan_rhd_functions import get_probe_name, read_header
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import struct
import datetime
import json


def create_eeg(filename, data, Fs, set_filename, scalar16, DC_Blocker=True, notch_freq=60, self=None):
    # data is given in int16

    if os.path.exists(filename):
        msg = '[%s %s]: The following EEG filename already exists: %s!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8], filename)

        if self is not None:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)
        return

    Fs_EGF = 4.8e3
    Fs_EEG = 250

    '''
    if DC_Blocker:
        data = filt.dcblock(data, 0.1, Fs_tint)
    '''

    msg = '[%s %s]: Filtering to create the EEG data!' % \
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8])

    if self is not None:
        self.LogAppend.myGUI_signal_str.emit(msg)
    else:
        print(msg)

    # LP at 500
    data = filt.iirfilt(bandtype='low', data=data, Fs=Fs, Wp=500, order=6,
                        automatic=0, Rp=0.1, As=60, filttype='cheby1', showresponse=0)

    # notch filter the data
    if notch_freq != 0:

        msg = '[%s %s]: Notch Filtering the EEG data!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8])

        if self is not None:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        data = filt.notch_filt(data, Fs, freq=notch_freq, band=10,
                               order=2, showresponse=0)
    else:

        msg = '[%s %s]: Notch Filtering the EEG data at a default of 60 Hz!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8])

        if self is not None:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        data = filt.notch_filt(data, Fs, freq=60, band=10,
                               order=2, showresponse=0)

    # downsample to 4.8khz signal for EGF signal (EEG is derived from EGF data)

    msg = '[%s %s]: Downsampling the EEG data to 4.8kHz!' % \
          (str(datetime.datetime.now().date()),
           str(datetime.datetime.now().time())[:8])

    # data = scipy.signal.decimate(data, int(Fs_tint / Fs_EGF), ftype='fir', axis=1)
    # data = scipy.signal.decimate(data, int(Fs_tint / Fs_EGF), axis=1)
    data = data[:, 0::int(Fs / Fs_EGF)]

    # notch filter the data
    # data = sp.Filtering().notch_filt(data, Fs_EGF, freq=60, band=10, order=3)

    # append zeros to make the duration a round number
    # duration_round = np.ceil(data.shape[1] / Fs_EGF)  # the duration should be rounded up to the nearest integer
    duration_round = int(get_setfile_parameter('duration', set_filename))  # get the duration from the set file
    missing_samples = int(duration_round * Fs_EGF - data.shape[1])

    if missing_samples != 0:
        missing_samples = np.tile(np.array([0]), (1, missing_samples))
        data = np.hstack((data, missing_samples))

    # data = np.rint(data)  # convert the data to integers
    # data = data.astype(np.int32)  # convert the data to integers
    # converting the data from uV to int16
    data = (data / scalar16).astype(np.int16)

    # ensuring the appropriate range of the values
    data[np.where(data > 32767)] = 32767
    data[np.where(data < -32768)] = -32768

    msg = '[%s %s]: Downsampling the EEG data to 250 Hz!' % \
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8])

    if self is not None:
        self.LogAppend.myGUI_signal_str.emit(msg)
    else:
        print(msg)

    # now apply lowpass at 125 hz to prevent aliasing of EEG
    # this uses a 101 tap von hann filter @ 125 Hz
    data, N = fir_hann(data, Fs_EGF, 125, n_taps=101, showresponse=0)

    data = int16toint8(data)

    data = EEG_downsample(data)

    ##################################################################################################
    # ---------------------------Writing the EEG Data-------------------------------------------
    ##################################################################################################

    write_eeg(filename, data, Fs_EEG, set_filename=set_filename)


def EEG_downsample(EEG):
    """The EEG data is created from the EGF files which involves a 4.8k to 250 Hz conversion"""
    EEG = EEG.flatten()

    i = -1
    # i = 0

    # indices = [i]
    indices = []
    while i < len(EEG) - 1:
        indices.extend([(i + 19), (i + 19 * 2), (i + 19 * 3), (i + 19 * 4), (i + 19 * 4 + 20)])
        # indices.extend([(i+20), (i+20+19), (i+20+19*2), (i+20+19*3), (i+20+19*4)])
        i += (19 * 4 + 20)

    indices = np.asarray(indices)

    indices = indices[np.where(indices <= len(EEG) - 1)]

    return EEG[indices]


def write_eeg(filepath, data, Fs, set_filename=None):
    data = data.flatten()

    session_path, session_filename = os.path.split(filepath)

    tint_basename = os.path.splitext(session_filename)[0]

    if set_filename is None:
        set_filename = os.path.join(session_path, '%s.set' % tint_basename)

    header = get_set_header(set_filename)

    num_samples = int(len(data))

    if '.egf' in session_filename:

        # EEG_Fs = 4800
        egf = True

    else:

        # EEG_Fs = 250
        egf = False

    # if the duration before the set file was overwritten wasn't a round number, it rounded up and thus we need
    # to append values to the EEG (we will add 0's to the end)
    duration = int(get_setfile_parameter('duration', set_filename))  # get the duration from the set file

    EEG_expected_num = int(Fs * duration)

    if num_samples < EEG_expected_num:
        missing_samples = EEG_expected_num - num_samples
        data = np.hstack((data, np.zeros((1, missing_samples)).flatten()))
        num_samples = EEG_expected_num

    with open(filepath, 'w') as f:

        num_chans = 'num_chans 1'

        if egf:
            sample_rate = '\nsample_rate %d Hz' % (int(Fs))
            data = struct.pack('<%dh' % (num_samples), *[np.int(data_value) for data_value in data.tolist()])
            b_p_sample = '\nbytes_per_sample 2'
            num_EEG_samples = '\nnum_EGF_samples %d' % (num_samples)

        else:
            sample_rate = '\nsample_rate %d.0 hz' % (int(Fs))
            data = struct.pack('>%db' % (num_samples), *[np.int(data_value) for data_value in data.tolist()])
            b_p_sample = '\nbytes_per_sample 1'
            num_EEG_samples = '\nnum_EEG_samples %d' % (num_samples)

        eeg_p_position = '\nEEG_samples_per_position %d' % (5)

        start = '\ndata_start'

        if egf:
            write_order = [header, num_chans, sample_rate,
                           b_p_sample, num_EEG_samples, start]
        else:
            write_order = [header, num_chans, sample_rate, eeg_p_position,
                           b_p_sample, num_EEG_samples, start]

        f.writelines(write_order)

    with open(filepath, 'rb+') as f:
        f.seek(0, 2)
        f.writelines([data, bytes('\r\ndata_end\r\n', 'utf-8')])


def fir_hann(data, Fs, cutoff, n_taps=101, showresponse=0):

    # The Nyquist rate of the signal.
    nyq_rate = Fs / 2

    b = scipy.signal.firwin(n_taps, cutoff / nyq_rate, window='hann')

    a = 1.0
    # Use lfilter to filter x with the FIR filter.
    data = scipy.signal.lfilter(b, a, data)
    # data = scipy.signal.filtfilt(b, a, data)

    if showresponse == 1:
        w, h = scipy.signal.freqz(b, a, worN=8000)  # returns the requency response h, and the angular frequencies
        # w in radians/sec
        # w (radians/sec) * (1 cycle/2pi*radians) = Hz
        # f = w / (2 * np.pi)  # Hz

        plt.figure(figsize=(20, 15))
        plt.subplot(211)
        plt.semilogx((w / np.pi) * nyq_rate, np.abs(h), 'b')
        plt.xscale('log')
        plt.title('%s Filter Frequency Response')
        plt.xlabel('Frequency(Hz)')
        plt.ylabel('Gain [V/V]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(cutoff, color='green')

    return data, n_taps


def get_set_header(set_filename):
    with open(set_filename, 'r+') as f:
        header = ''
        for line in f:
            header += line
            if 'sw_version' in line:
                break
    return header


def create_egf(filename, data, Fs, set_filename, scalar16, DC_Blocker=True, notch_freq=60, self=None):

    if os.path.exists(filename):
        msg = '[%s %s]: The following EGF filename already exists: %s!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8], filename)

        if self is not None:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)
        return

    Fs_EGF = 4.8e3

    '''
    if DC_Blocker:
        data = filt.dcblock(data, 0.1, Fs_tint)
    '''

    msg = '[%s %s]: Filtering to create the EGF data!' % \
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8])

    if self is not None:
        self.LogAppend.myGUI_signal_str.emit(msg)
    else:
        print(msg)

    # LP at 500
    data = filt.iirfilt(bandtype='low', data=data, Fs=Fs, Wp=500, order=6,
                        automatic=0, Rp=0.1, As=60, filttype='cheby1', showresponse=0)

    # notch filter the data
    if notch_freq != 0:

        msg = '[%s %s]: Notch Filtering the EGF data!' % \
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8])

        if self is not None:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        data = filt.notch_filt(data, Fs, freq=notch_freq, band=10,
                               order=2, showresponse=0)
    else:

        msg = '[%s %s]: Notch Filtering the EGF data at a default of 60 Hz!' % \
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8])

        if self is not None:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        data = filt.notch_filt(data, Fs, freq=60, band=10,
                               order=2, showresponse=0)

    # downsample to 4.8khz signal for EGF signal (EEG is derived from EGF data)

    msg = '[%s %s]: Downsampling the EGF data to 4.8kHz!' % \
        (str(datetime.datetime.now().date()),
         str(datetime.datetime.now().time())[:8])

    # data = scipy.signal.decimate(data, int(Fs_tint / Fs_EGF), ftype='fir', axis=1)
    # data = scipy.signal.decimate(data, int(Fs_tint / Fs_EGF), axis=1)
    data = data[:, 0::int(Fs / Fs_EGF)]

    # notch filter the data
    # data = sp.Filtering().notch_filt(data, Fs_EGF, freq=60, band=10, order=3)

    # append zeros to make the duration a round number
    # duration_round = np.ceil(data.shape[1] / Fs_EGF)  # the duration should be rounded up to the nearest integer
    duration_round = int(get_setfile_parameter('duration', set_filename))  # get the duration from the set file
    missing_samples = int(duration_round * Fs_EGF - data.shape[1])
    if missing_samples != 0:
        missing_samples = np.tile(np.array([0]), (1, missing_samples))
        data = np.hstack((data, missing_samples))

    # data = np.rint(data)  # convert the data to integers
    # data = data.astype(np.int32)  # convert the data to integers
    # converting the data from uV to int16
    data = (data / scalar16).astype(np.int16)

    # ensuring the appropriate range of the values
    data[np.where(data > 32767)] = 32767
    data[np.where(data < -32768)] = -32768

    # data is already in int16 which is what the final unit should be in

    ##################################################################################################
    # ---------------------------Writing the EGF Data-------------------------------------------
    ##################################################################################################

    write_eeg(filename, data, Fs_EGF, set_filename=set_filename)


def eeg_channels_to_filenames(channels, directory, output_basename):
    eeg_filenames = []
    egf_filenames = []

    for i in np.arange(len(channels)):
        eeg_number = i + 1

        if eeg_number == 1:
            eeg_filename = os.path.join(directory, output_basename + '.eeg')
            egf_filename = os.path.join(directory, output_basename + '.egf')
        else:
            eeg_filename = os.path.join(directory, output_basename + '.eeg%d' % (eeg_number))
            egf_filename = os.path.join(directory, output_basename + '.egf%d' % (eeg_number))

        eeg_filenames.append(eeg_filename)
        egf_filenames.append(egf_filename)

    return eeg_filenames, egf_filenames


def get_eeg_channels(probe_map, directory, output_basename, channels='all'):
    """

    :param method:
    :return:
    """

    tetrode_channels = probe_map.values()

    if type(channels) == str:
        if channels == 'all':
            # All of the channels will be saved as an EEG / EGF file
            channels = np.asarray(list(tetrode_channels)).flatten()
        elif channels == 'first':
            # only convert the first channel in each tetrode
            channels = np.asarray([channel[0] for channel in tetrode_channels])

    eeg_filenames, egf_filenames = eeg_channels_to_filenames(channels, directory, output_basename)

    return eeg_filenames, egf_filenames, channels


def convert_eeg(session_files, tint_basename, output_basename, Fs, convert_channels='first', self=None):
    """
    This method will create all the eeg and egf files


    """
    directory = os.path.dirname(session_files[0])

    raw_fnames = [os.path.join(directory, file) for file in os.listdir(
        directory) if '_raw.mda' in file if os.path.basename(tint_basename) in file]

    probe = get_probe_name(session_files[0])

    probe_map = tetrode_map[probe]

    # get eeg and egf files + channels to convert
    eeg_filenames, egf_filenames, eeg_channels = get_eeg_channels(probe_map, directory, output_basename,
                                                                  channels=convert_channels)

    total_files_n = len(eeg_filenames) + len(egf_filenames)

    cue_fname = os.path.join(directory, tint_basename + '_cues.json')

    with open(cue_fname) as f:
        cue_data = json.load(f)

    if 'converted_eeg' in cue_data.keys():
        # then the session has been converted already, check to ensure that we maintain the correct naming

        # get the converted eeg channel numbers
        converted_eegs_dict = cue_data['converted_eeg']

        converted_eegs = np.asarray([converted_eegs_dict[key] for key in sorted(
            converted_eegs_dict)])

        # get the converted eeg/egf filenames
        eeg_converted, egf_converted = eeg_channels_to_filenames(converted_eegs, directory, output_basename)
        for i in np.arange(len(eeg_converted)):
            if i < len(eeg_channels):
                if converted_eegs[i] != eeg_channels[i]:
                    # then the eeg_filename corresponds to the wrong channel, delete it
                    os.remove(eeg_converted[i])
                    os.remove(egf_converted[i])
            else:
                # delete these files, they are no longer necessary in the new conversion
                os.remove(eeg_converted[i])
                os.remove(egf_converted[i])

    else:
        # deleting old files before the implementation of converted_eeg was used
        eeg_converted = [os.path.join(directory, file) for file in os.listdir(directory) if
                         output_basename + '.eeg' in file]
        egf_converted = [os.path.join(directory, file) for file in os.listdir(directory) if
                         output_basename + '.egf' in file]

        for file in eeg_converted + egf_converted:
            os.remove(file)

    # check if the files have already been created
    file_exists = 0
    for create_file in (eeg_filenames + egf_filenames):
        if os.path.exists(create_file):
            file_exists += 1

    if file_exists == total_files_n:

        msg = '[%s %s]: All the EEG/EGF files for the following session is converted: %s!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8], tint_basename)
        if self is not None:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        return False

    # set_filename = '%s.set' % (os.path.join(directory, tint_basename))
    set_filename = '%s.set' % output_basename

    for file in raw_fnames:

        mda_basename = os.path.splitext(file)[0]
        mda_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

        tint_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

        tetrode = int(mda_basename[find_sub(mda_basename, '_')[-1] + 2:])

        tetrode_channels = probe_map[tetrode]

        channel_bool = np.where(np.in1d(eeg_channels, tetrode_channels))[0]

        if len(channel_bool) > 0:
            # then this tetrode has a channel to be saved
            current_files = [eeg_filenames[x] for x in channel_bool] + \
                            [egf_filenames[x] for x in channel_bool]

            file_written = 0
            for cfile in current_files:
                if os.path.exists(cfile):
                    file_written += 1

            if file_written == len(current_files):
                for cfile in current_files:
                    if '.eeg' in file:
                        msg = '[%s %s]: The following EEG file has already been created, skipping: %s!' % \
                              (str(datetime.datetime.now().date()),
                               str(datetime.datetime.now().time())[:8], cfile)
                    else:
                        msg = '[%s %s]: The following EGF file has already been created, skipping: %s!' % \
                              (str(datetime.datetime.now().date()),
                               str(datetime.datetime.now().time())[:8], cfile)
                    if self is not None:
                        self.LogAppend.myGUI_signal_str.emit(msg)
                    else:
                        print(msg)
                continue

            data, _ = readMDA(file)  # size: len(tetrode_channels) x number of samples, units: bits

            # convert from bits to uV
            data = data * intan_scalar()  # units: uV

            file_header = read_header(session_files[0])  # read the file header information from a session file
            n_channels = file_header['num_amplifier_channels']

            clip_filename = '%s_clips.json' % (os.path.join(directory, tint_basename))

            if os.path.exists(clip_filename):
                with open(clip_filename, 'r') as f:
                    clip_data = json.load(f)

                channel_scalar16s = np.zeros(n_channels)

                for tetrode in probe_map.keys():
                    tetrode_clips = clip_data[str(tetrode)]
                    for channel in probe_map[tetrode]:
                        channel_scalar16s[channel - 1] = tetrode_clips['scalar16bit']
            else:
                raise FileNotFoundError('Clip Filename not found!')

            # for eeg_filename, egf_filename, channel_number in zip(eeg_filenames, egf_filenames, probe_map[tetrode]):
            for i in channel_bool:
                eeg_filename = eeg_filenames[i]
                egf_filename = egf_filenames[i]
                channel_number = eeg_channels[i]
                channel_i = np.where(tetrode_channels == channel_number)[0]

                if os.path.exists(eeg_filename):

                    EEG = np.array([])
                    msg = '[%s %s]: The following EEG file has already been created, skipping: %s!' % \
                          (str(datetime.datetime.now().date()),
                           str(datetime.datetime.now().time())[:8], eeg_filename)
                    if self is not None:
                        self.LogAppend.myGUI_signal_str.emit(msg)
                    else:
                        print(msg)
                else:
                    msg = '[%s %s]: Creating the following EEG file: %s!' % \
                          (str(datetime.datetime.now().date()),
                           str(datetime.datetime.now().time())[:8], eeg_filename)

                    if self is not None:
                        self.LogAppend.myGUI_signal_str.emit(msg)
                    else:
                        print(msg)

                    # load data
                    EEG = data[channel_i, :].reshape((1, -1))

                    create_eeg(eeg_filename, EEG, Fs, set_filename, channel_scalar16s[channel_number - 1],
                               DC_Blocker=False, self=self)

                    # this will overwrite the eeg settings that is in the .set file
                    eeg_ext = os.path.splitext(eeg_filename)[1]
                    if eeg_ext == '.eeg':
                        eeg_number = '1'
                    else:
                        eeg_number = eeg_ext[4:]

                if os.path.exists(egf_filename):

                    msg = '[%s %s]: The following EGF file has already been created, skipping: %s!' % \
                          (str(datetime.datetime.now().date()),
                           str(datetime.datetime.now().time())[:8], egf_filename)
                    if self is not None:
                        self.LogAppend.myGUI_signal_str.emit(msg)
                    else:
                        print(msg)

                else:

                    msg = '[%s %s]: Creating the following EGF file: %s!' % \
                          (str(datetime.datetime.now().date()),
                           str(datetime.datetime.now().time())[:8], egf_filename)

                    if self is not None:
                        self.LogAppend.myGUI_signal_str.emit(msg)
                    else:
                        print(msg)

                    EEG = data[channel_i, :].reshape((1, -1))

                    create_egf(egf_filename, EEG, Fs, set_filename, channel_scalar16s[channel_number - 1],
                               DC_Blocker=False, self=self)

                EEG = None

    eeg_dict = {}
    for k in np.arange(len(eeg_channels)):
        # using the int because json doesn't like np.int32
        eeg_dict[int(k)] = int(eeg_channels[k])

    cue_data['converted_eeg'] = eeg_dict

    with open(cue_fname, 'w') as f:
        json.dump(cue_data, f)

    return True
