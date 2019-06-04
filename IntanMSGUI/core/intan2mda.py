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
    # rank contains the index values of the input array that would create the sorted array.
    # i.e. rank = [2,0,1] means that the 0'th index of the sorted array should be the 2nd
    # index of the input array the 1st index of the sorted array will be the 0'th input of
    # the sorted array, etc.

    # order contains the index values where the value of the given array would fit into the sorted
    # array i.e. order = [1,2,0] means that the 0'th index of the input array should be the 1st
    # index of the sorted array the 1st index of the input array will be the 2nd input of the
    # sorted array, etc.

    ranking = np.argsort(values)[::-1]

    order = np.r_[[np.where(value == ranking)[0]
                   for value in np.arange(len(values))]].flatten()

    if mode == 'ascending':
        order = np.flip(order)
        ranking = np.flip(ranking)

    return ranking, order


def get_reref_data(session_files, probe_map, channel=None, mode='sd', start_index=None, stop_index=None):
    """
    This will get channel data for software re-referencing. Typically this will remove common data between the different
    channels, such as that which could be caused from motion artifact. Likely you will want to choose a channel with
    small amounts of variation and low spiking activity.

    if mode is 'sd':
        In the common mode removal process we will take the signal with the smallest standard deviation
        and subtract it from all the other channels. We will also find the signal with the 2nd smallest std
        and remove this channel from the channel with the smallest standard deviation. This function will find both
        those channels

    if mode is 'avg':
        In the common mode removal process we will take the average of all the channels and use this value to subtract
        the common signal from each of the channels.

    start_index and stop_index are used to sync the ephys with the behavior. It will be the index value
    that represents the start and end of the behavior
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

    if channel is None:
        # read each tetrode and calculate the avg/sd depending on the mode
        for tetrode, tetrode_channels in sorted(probe_map.items()):
            channel_list.extend(tetrode_channels)  # keep track of which channels were

            if mode == 'sd' or mode == 'avg':
                data, _, _, _ = get_intan_data(session_files, tetrode_channels, tetrode,
                                               analog_data=False, digital_data=False, ephys_data=True)

                if start_index is not None and stop_index is not None:
                    data = data[:, start_index:stop_index]

                if mode == 'sd':
                    # calculate standard deviations for each channel as a measure of how quiet the channel is
                    if len(channel_std) == 0:
                        channel_std = np.std(data, axis=1)
                    else:
                        channel_std = np.r_[channel_std, np.std(data, axis=1)]
                elif mode == 'avg':
                    # calculate the averages for the channel, we'll sum them here and divide by n_channels later
                    num_channels += data.shape[0]

                    if len(channel_avg) == 0:
                        channel_avg = np.sum(data, axis=0)
                    else:
                        channel_avg += np.sum(data, axis=0)
                data = None

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
            reref_channels = channel_list[rank[:2]]

            data, _, _, _ = get_intan_data(session_files, reref_channels, verbose=None,
                                           analog_data=False, digital_data=False, ephys_data=True)

            # only include the appropriate data where the behavior was running
            if start_index is not None and stop_index is not None:
                data = data[:, start_index:stop_index]

            else:
                # then there is no start and stop signal, just bind the data so that it is at an even numbered duration
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

            return data, reref_channels

        elif mode == 'avg':

            # converting the channel sums into channel averages
            channel_avg = channel_avg / num_channels

            return channel_avg

    else:
        # the channels were chosen

        if type(channel) is not list:
            channel = [channel]

        data, _, _, _ = get_intan_data(session_files, channel, verbose=None,
                                       analog_data=False, digital_data=False, ephys_data=True)

        # only include the appropriate data where the behavior was running
        if start_index is not None and stop_index is not None:
            data = data[:, start_index:stop_index]

        else:
            # there were no start and stop signals

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
              flip_sign=True, software_rereference=False, reref_channels=None, reref_data=None,
              reref_method=None, start_index=None, stop_index=None, self=None):
    """
        This function convert the intan ata into the .mda format that MountainSort requires.

        Example:
            session_files = ['C:\\example\\example_session_1.rhd' , 'C:\\example\\example_session_2.rhd']
            directory = 'C:\\example'
            intan2mda(session_files, directory)

        Args:
            tetrode_files (list): list of the fullpaths to the tetrode files belonging to the session you want to
                convert.
            desired_Fs (float): the sampling rate that you want the data to be at. I've added this because we generally
                record at 24k but we will convert to a 48k signal (the system we analyze w/ needs 48k).
            directory (string): the path of the directory the session and tetrode files are in.
            basename (string): the basename of the session you are converting.
            pre_threshold (int): the number of samples before the threshold that you want to save in the waveform
            post_threshold (int): the number of samples after the threshold that you want to save in the waveform
            start_index (int): This will represent the index of the ephys data that the behavior started
            stop_index (int): This will represent the idnex of the ephys data that the behavior ended
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

    # determine if the files have already been converted
    for tetrode, tetrode_channels in sorted(probe_map.items()):

        mda_filename = '%s_T%d_raw.mda' % (os.path.join(directory, tint_basename), tetrode)

        if os.path.exists(mda_filename):
            converted_files += 1

        n_tetrodes += 1

    if n_tetrodes == converted_files:
        # all the tetrodes have been converted, return the recording duration
        msg = '[%s %s]: All the tetrodes have already been created, skipping!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8])
        if self:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)
        # getting the duration
        for tetrode, tetrode_channels in sorted(probe_map.items()):
            _, t_intan, _, _ = get_intan_data(session_files, tetrode_channels, tetrode, self,
                                              verbose=True, analog_data=False, digital_data=False, ephys_data=False)

            if start_index is not None and stop_index is not None:
                duration = int((stop_index - start_index) / Fs_intan)

            else:
                # there was no start or stop index
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
            return duration

    # convert each tetrode
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
        data, t_intan, _, _ = get_intan_data(session_files, tetrode_channels, tetrode, self,
                                             verbose=True, analog_data=False, digital_data=False, ephys_data=True)

        # splice the ephys data so that it only occurred between the start and stop of the maze
        # (info that is in the digital signal)

        if start_index is not None and stop_index is not None:
            duration = int((stop_index - start_index) / Fs_intan)
            # slicing the data to make it bound between the start and (new) stop
            # of the behavior
            data = data[:, start_index:stop_index]
            t_intan = t_intan[start_index:stop_index] - t_intan[start_index]

        else:
            # then there was no start and stop signal in the analog / digital
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

        # software re-referencing
        if software_rereference:
            # perform software re-referencing, to remove any common data between the tetrodes / channels
            tetrode_channels = np.asarray(tetrode_channels)
            reref_channels = np.asarray(reref_channels)
            if reref_method == 'sd':
                # with this remove common mode process we will take the signal with the smallest standard deviation
                # and subtract it from all the other channels. We will also find the signal with the 2nd smallest
                # std and remove this channel from the channel with the smallest standard deviation
                if reref_channels[0] in tetrode_channels:
                    # find if the re-referencing channel is in this tetrode
                    channel_bool = np.where(tetrode_channels == reref_channels[0])[0]
                    # subtract the alternate channel from this
                    data[channel_bool, :] -= reref_data[1, :]

                # subtract any channels that are not equal to the smallest SD channel
                channel_bool = np.where(tetrode_channels != reref_channels[0])[0]
                data[channel_bool, :] -= reref_data[0, :]

            elif reref_method == 'avg':
                # calculate the avg of the channels and then subtract that from each of the channels
                data -= reref_data.astype(np.int32)

            elif reref_method is None:
                # then we gave the channels
                if len(reref_channels) == 2:
                    if reref_channels[0] in tetrode_channels:
                        # find if the re-referencing channel is in this tetrode
                        channel_bool = np.where(tetrode_channels == reref_channels[0])[0]
                        # subtract the alternate ref channel from this
                        data[channel_bool, :] -= reref_data[1, :]

                    # subtract any channels that are not equal to the primary reference channel
                    channel_bool = np.where(tetrode_channels != reref_channels[0])[0]
                    data[channel_bool, :] -= reref_data[0, :]
                elif len(reref_channels) == 1:
                    # there is only one reference channel given, we will just use this to subtract from all
                    # the other channels. We will leave the reference channel alone though

                    # subtract any channels that are not equal to the reference channel
                    channel_bool = np.where(tetrode_channels != reref_channels[0])[0]
                    data[channel_bool, :] -= reref_data[0, :]

        # notch filter the signal
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

                if desired_Fs != 48e3 and desired_Fs != 24e3:
                    msg = '[%s %s]: For Tint you need to record at 48k or 24k, you chose %.2f!' % \
                          (str(datetime.datetime.now().date()),
                           str(datetime.datetime.now().time())[:8], float(desired_Fs))
                    if self:
                        self.LogAppend.myGUI_signal_str.emit(msg)
                    else:
                        print(msg)

                    raise ValueError('Improper desired_Fs: %.2f' % float(desired_Fs))

                msg = '[%s %s]: Current Fs is %d, we want a Fs of %d, interpolating!' % \
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

        data = np.int16(data)

        _writemda(data, mda_filename, 'int16')

    return duration
