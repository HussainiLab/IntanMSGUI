from core.intan_rhd_functions import read_header, get_data_limits, get_probe_name, get_intan_data
import numpy as np
import os
import datetime
import math
from core.rhd_utils import tetrode_map
from core.filtering import notch_filt
from scipy import interpolate
from core.mountainsort_functions import _writemda


def compute_ranking(values, mode='descending'):
    if mode == 'descending':
        ranking = np.argsort(values)[::-1]
    elif mode == 'ascending':
        ranking = np.argsort(values)

    order = np.r_[[np.where(value == ranking)[0]
                   for value in np.arange(len(values))]].flatten()

    return ranking, order


def get_common_mode(session_files, probe_map, channel=None, mode='sd'):
    """
    This will remove the common noise that exists across all the channels.

    if mode is 'sd':
        In the common mode removal process we will take the signal with the smallest standard deviation
        and subtract it from all the other channels. We will also find the signal with the 2nd smallest std
        and remove this channel from the channel with the smallest standard deviation. This function will find both
        those channels

    if mode is 'avg':
        In the common mode removal process we will take the average of all the channels and use this value to subtract
        the commond signal from each of the channels.
    """
    # keep track of the channels
    channel_list = []

    # keep track of the standard deviations
    channel_std = np.array([])
    channel_avg = np.array([])
    num_channels = 0

    file_header = read_header(session_files[0])

    Fs_intan = file_header['frequency_parameters']['amplifier_sample_rate']
    if type(Fs_intan) == np.ndarray:
        Fs_intan = Fs_intan[0]

    tint_basename = os.path.basename(os.path.splitext(sorted(session_files, reverse=False)[0])[0])
    directory = os.path.dirname(session_files[0])

    if channel is None:
        for tetrode, tetrode_channels in sorted(probe_map.items()):
            channel_list.extend(tetrode_channels)  # keep track of which channels were
            data, _, data_digital_in = get_intan_data(session_files, tetrode_channels, tetrode)

            if file_header['num_board_dig_in_channels'] > 0:
                start_index, stop_index = get_data_limits(directory, tint_basename, data_digital_in)

                if start_index is not None and stop_index is not None:
                    data = data[:, start_index:stop_index]

            if mode == 'sd':
                if len(channel_std) == 0:
                    channel_std = np.std(data, axis=1)
                else:
                    channel_std = np.r_[channel_std, np.std(data, axis=1)]
            elif mode == 'avg':
                num_channels += data.shape[0]

                if len(channel_avg) == 0:
                    channel_avg = np.sum(data, axis=0)
                else:
                    channel_avg += np.sum(data, axis=0)

            data = None
            data_digital_in = None

        if mode == 'sd':

            channel_list = np.asarray(channel_list)

            # rank contains the index values of the input array that would create the sorted array.
            # i.e. rank = [2,0,1] means that the 0'th index of the sorted array should be the 2nd index of the input
            # array the 1st index of the sorted array will be the 0'th input of the sorted array, etc.

            # order contains the index values where the value of the given array would fit into the sorted array
            # i.e. order = [1,2,0] means that the 0'th index of the input array should be the 1st index of the sorted
            # array the 1st index of the input array will be the 2nd input of the sorted array, etc.

            rank, order = compute_ranking(channel_std, mode='ascending')

            # find the channels with the two lowest standard deviations. The lowest will be subtracted from all the
            # other channels and the 2nd lowest will be subtracted from the channel with the 1st lowest.
            mode_channels = channel_list[rank[:2]]

            data, _, data_digital_in = get_intan_data(session_files, mode_channels, verbose=None)

            digital_input = False
            # check if there is a start and stop index from the recorded digital events
            if file_header['num_board_dig_in_channels'] > 0:
                start_index, stop_index = get_data_limits(directory, tint_basename, data_digital_in)

                # only include the appropriate data where the behavior was running
                if start_index is not None and stop_index is not None:
                    duration = np.floor((stop_index - start_index) / Fs_intan)
                    # we floor the value so we can have a integer value for the duration
                    n = int(duration * Fs_intan)

                    # creating a new stop_index with the new number of samples that will
                    # allow for the rounded integer value duration
                    stop_index = start_index + n

                    digital_input = True
                    data = data[:, start_index:stop_index]

            if not digital_input:

                duration = data.shape[1] / Fs_intan  # duration value that would be reported in the .pos/.set settings
                if type(duration) == np.ndarray:
                    duration = duration[0]  # the following is_integer() method does not work in an ndarray

                if not duration.is_integer():
                    '''
                    Tint needs integer values in the duration spot of the .pos file,
                    so we will round down if not an integer. This will likely be fixed
                    if the start and stop indices were found via the digital inputs. However 
                    if that failed, or if there is no digital input to sync with the behavior
                    we will implement this if statement.
                    '''
                    duration = math.floor(duration)  # the new integer duration value
                    n = int(duration * Fs_intan)

                    data = data[:, :n]

            return data, mode_channels, channel_list

        elif mode == 'avg':

            # converting the channel sums into channel averages
            channel_avg = channel_avg / num_channels

            return channel_avg

    else:

        if type(channel) is not list:
            channel = [channel]

        data, _, data_digital_in = get_intan_data(session_files, channel, verbose=None)
        # check if there is a start and stop index from the recorded digital events
        digital_input = False
        if file_header['num_board_dig_in_channels'] > 0:
            start_index, stop_index = get_data_limits(directory, tint_basename, data_digital_in)

            # only include the appropriate data where the behavior was running
            if start_index is not None and stop_index is not None:
                duration = np.floor((stop_index - start_index) / Fs_intan)
                # we floor the value so we can have a integer value for the duration
                n = int(duration * Fs_intan)

                # creating a new stop_index with the new number of samples that will
                # allow for the rounded integer value duration
                stop_index = start_index + n

                digital_input = True
                data = data[:, start_index:stop_index]

        if not digital_input:

            duration = data.shape[1] / Fs_intan  # duration value that would be reported in the .pos/.set settings
            if type(duration) == np.ndarray:
                duration = duration[0]  # the following is_integer() method does not work in an ndarray

            if not duration.is_integer():
                '''
                Tint needs integer values in the duration spot of the .pos file,
                so we will round down if not an integer. This will likely be fixed
                if the start and stop indices were found via the digital inputs. However 
                if that failed, or if there is no digital input to sync with the behavior
                we will implement this if statement.
                '''
                duration = math.floor(duration)  # the new integer duration value
                n = int(duration * Fs_intan)

                data = data[:, :n]

        return data, np.asarray(channel)


def intan2mda(session_files, desired_Fs=48e3, interpolation=True, notch_filter=True, notch_freq=60,
              flip_sign=True, remove_common_mode=True, common_mode_method='sd', common_mode_channels=None, self=None):
    """
        This function convert the intan ata into the .mda format that MountainSort requires.

        Example:
            session_files = ['C:\\example\\example_session_1.rhd' , 'C:\\example\\example_session_2.rhd']
            directory = 'C:\\example'
            intan2mda(session_files, directory)

        Args:
            tetrode_files (list): list of the fullpaths to the tetrode files belonging to the session you want to
                convert.
            desired_Fs (int): the sampling rate that you want the data to be at. I've added this because we generally
                record at 24k but we will convert to a 48k signal (the system we analyze w/ needs 48k).
            directory (string): the path of the directory the session and tetrode files are in.
            basename (string): the basename of the session you are converting.
            pre_threshold (int): the number of samples before the threshold that you want to save in the waveform
            post_threshold (int): the number of samples after the threshold that you want to save in the waveform
            self (object): self is the main window of a GUI that has a LogAppend method that will append this message to
                a log (QTextEdit), and a custom signal named myGUI_signal_str that would was hooked up to LogAppend to
                append messages to the log.
        Returns:
            mdas_written (list): a list of mda files that were created
    """

    duration = None

    file_header = read_header(session_files[0])

    Fs_intan = file_header['frequency_parameters']['amplifier_sample_rate']
    if type(Fs_intan) == np.ndarray:
        Fs_intan = Fs_intan[0]

    # get the probe value from the notes
    probe = get_probe_name(session_files[0])

    probe_map = tetrode_map[probe]

    tint_basename = os.path.basename(os.path.splitext(sorted(session_files, reverse=False)[0])[0])

    directory = os.path.dirname(session_files[0])

    mda_filenames = []

    converted_files = 0
    n_tetrodes = 0
    for tetrode, tetrode_channels in sorted(probe_map.items()):

        mda_filename = '%s_T%d_raw.mda' % (os.path.join(directory, tint_basename), tetrode)

        if os.path.exists(mda_filename):
            converted_files += 1

        n_tetrodes += 1

    if n_tetrodes == converted_files:
        msg = '[%s %s]: All the tetrodes have already been created, skipping!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8])
        if self:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)
        return [], []

    if remove_common_mode:

        msg = '[%s %s]: Finding common mode data!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8])
        if self:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        if common_mode_channels is None:
            if common_mode_method == 'sd':
                common_mode_data, mode_channels = get_common_mode(session_files, probe_map, mode=common_mode_method)
            elif common_mode_method == 'avg':
                common_mode_data = get_common_mode(session_files, probe_map, mode=common_mode_method)
        else:
            common_mode_data, mode_channels = get_common_mode(session_files, probe_map, channel=common_mode_channels)

    # find the common mode

    for tetrode, tetrode_channels in sorted(probe_map.items()):

        msg = '[%s %s]: Converting the following tetrode: %d!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8], tetrode)
        if self:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        mda_filename = '%s_T%d_raw.mda' % (os.path.join(directory, tint_basename), tetrode)

        mda_filenames.append(mda_filename)

        if os.path.exists(mda_filename):
            msg = '[%s %s]: The following filename already exists: %s, skipping conversion!#Red' % \
                  (str(datetime.datetime.now().date()),
                   str(datetime.datetime.now().time())[:8], mda_filename)

            if self:
                self.LogAppend.myGUI_signal_str.emit(msg)
            else:
                print(msg)
            continue

        # get_tetrode_data
        data, t_intan, data_digital_in = get_intan_data(session_files, tetrode_channels, tetrode, self, verbose=True)

        # splice the ephys data so that it only occured between the start and stop of the maze
        # (info that is in the digital signal)

        digital_inputs = False
        if file_header['num_board_dig_in_channels'] > 0:
            start_index, stop_index = get_data_limits(directory, tint_basename, data_digital_in)

            if start_index is not None and stop_index is not None:
                duration = np.floor((stop_index - start_index) / Fs_intan)
                # we floor the value so we can have a integer value for the duration
                n = int(duration * Fs_intan)

                # creating a new stop_index with the new number of samples that will
                # allow for the rounded integer value duration
                stop_index = start_index + n

                # slicing the data to make it bound between the start and (new) stop
                # of the behavior
                data = data[:, start_index:stop_index]
                t_intan = t_intan[start_index:stop_index] - t_intan[start_index]

                digital_inputs = True

        if not digital_inputs:
            duration = len(t_intan) / Fs_intan  # duration value that would be reported in the .pos/.set settings
            if type(duration) == np.ndarray:
                duration = duration[0]  # the following is_integer() method does not work in an ndarray

            if not duration.is_integer():
                '''
                Tint needs integer values in the duration spot of the .pos file,
                so we will round down if not an integer. This will likely be fixed
                if the start and stop indices were found via the digital inputs. However 
                if that failed, or if there is no digital input to sync with the behavior
                we will implement this if statement.
                '''
                duration = math.floor(duration)  # the new integer duration value
                n = int(duration * Fs_intan)

                data = data[:, :n]
                t_intan = t_intan[:n] - t_intan[0]

        if remove_common_mode:

            if common_mode_channels is None:
                if common_mode_method == 'sd' or common_mode_channels is not None:
                    # with this remove common mode process we will take the signal with the smallest standard deviation
                    # and subtract it from all the other channels. We will also find the signal with the 2nd smallest
                    # std and remove this channel from the channel with the smallest standard deviation
                    if mode_channels[0] in tetrode_channels:
                        channel_bool = np.where(tetrode_channels == mode_channels[0])[0]
                        data[channel_bool, :] -= common_mode_data[1, :]

                    channel_bool = np.where(tetrode_channels != mode_channels[0])[0]
                    data[channel_bool, :] -= common_mode_data[0, :]

                elif common_mode_method == 'avg':
                    # calculate the avg of the channels and then subtract that from each of the channels
                    data -= common_mode_data.astype(np.int32)
            else:
                if common_mode_data.shape[0] == 2:
                    if mode_channels[0] in tetrode_channels:
                        channel_bool = np.where(tetrode_channels == mode_channels[0])[0]
                        data[channel_bool, :] -= common_mode_data[1, :]

                    channel_bool = np.where(tetrode_channels != mode_channels[0])[0]
                    data[channel_bool, :] -= common_mode_data[0, :]
                else:
                    data -= common_mode_data.astype(np.int32)

        if notch_filter:
            data = notch_filt(data, Fs_intan, freq=notch_freq)

        # flip the signal because Tint likes positive peaks
        if flip_sign:
            data = np.multiply(-1, data)
            # because we flipped the signs we need to make sure that it is still int16
            data[np.where(data >= 32767)] = 32767

        if interpolation:
            # interpolate if we need to
            if Fs_intan != desired_Fs:

                msg = '[%s %s]: Current Fs is %d, we want a Fs of %d, interpolating!!' % \
                      (str(datetime.datetime.now().date()),
                       str(datetime.datetime.now().time())[:8], int(Fs_intan), int(desired_Fs))
                if self:
                    self.LogAppend.myGUI_signal_str.emit(msg)
                else:
                    print(msg)

                # we will interpolate
                n_samples = int(duration * desired_Fs)

                t = np.arange(n_samples) / desired_Fs  # creates a time array of the signal starting from 0

                interp_function = interpolate.interp1d(t_intan, data, kind='linear', fill_value="extrapolate", axis=1)

                data = interp_function(t)

        else:
            pass

        data = np.int16(data)

        _writemda(data, mda_filename, 'int16')

    return duration