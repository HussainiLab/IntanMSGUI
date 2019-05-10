import os
import datetime
import numpy as np
from core.tetrode_conversion import batch_basename_tetrodes, batch_add_tetrode_headers
from core.convert_position import convert_position
from core.eeg_conversion import convert_eeg, get_eeg_channels
from core.utils import find_sub
from core.intan2mda import intan2mda, get_reref_data
from core.mdaSort import sort_intan
from core.set_conversion import convert_setfile, overwrite_eeg_set_params
from core.rhd_utils import tetrode_map
from core.intan_rhd_functions import read_header, get_probe_name, get_ref_index, get_data_limits, read_data, read_header


def validate_session(rhd_basename_file, output_basename, convert_channels, self=None, verbose=True, require_pos=True):
    """
    This will return an output of True if you should continue to convert this session,
    otherwise it is convertable.
    """

    output_basename = os.path.basename(output_basename)

    # get the probe value from the notes
    probe = get_probe_name(rhd_basename_file)

    probe_map = tetrode_map[probe]

    tint_basename = os.path.basename(os.path.splitext(rhd_basename_file)[0])
    directory = os.path.dirname(rhd_basename_file)

    # first check if this session has the necessary files

    pos_filename = '%s.pos' % os.path.join(directory, output_basename)
    # cues_filename = '%s_cues.json' % os.path.join(directory, tint_basename)
    raw_pos_filename = '%s_raw_position.txt' % os.path.join(directory, tint_basename)

    # check if there is a position file

    if require_pos:
        if not os.path.exists(pos_filename):
            # there is no position filename, check if there is a raw position file to
            # create a position file with.

            if not os.path.exists(raw_pos_filename):
                if verbose:
                    msg = ('[%s %s]: There is no .pos file or raw position file to create a ' + \
                           '.pos file! Skipping the following basename: %s!') % \
                          (str(datetime.datetime.now().date()),
                           str(datetime.datetime.now().time())[:8], tint_basename)
                    if self:
                        self.LogAppend.myGUI_signal_str.emit(msg)
                    else:
                        print(msg)
                return False

    # TODO: Add a check for the cues_filename, will add it if we need to.

    # check that all the files haven't already been converted
    
    # check that all tetrodes haven't already been created
    converted_files = 0
    n_tetrodes = 0
    for tetrode, tetrode_channels in sorted(probe_map.items()):

        # mda_filename = '%s_T%d_raw.mda' % (os.path.join(directory, tint_basename), tetrode)
        mda_filename = '%s_T%d_firings.mda' % (os.path.join(directory, tint_basename), tetrode)

        if os.path.exists(mda_filename):
            converted_files += 1

        n_tetrodes += 1

    if n_tetrodes != converted_files:
        return True

    # check that all the tetrodes have been converted

    # raw_fnames = [os.path.join(directory, file) for file in os.listdir(
    #     directory) if '_raw.mda' in file if tint_basename in file]

    firing_fnames = [os.path.join(directory, file) for file in os.listdir(
         directory) if '_firings.mda' in file if tint_basename in file]

    # for file in raw_fnames:
    for file in firing_fnames:
        mda_basename = os.path.splitext(file)[0]
        mda_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

        # I used to check all of these files however, I added this cleanup function that will delete the unnecessary
        # ones, so we delete most of this besides the firings and the metrics.

        # masked_out_fname = mda_basename + '_masked.mda'
        # filt_out_fname = mda_basename + '_filt.mda'
        # pre_out_fname = mda_basename + '_pre.mda'

        # firings_out = mda_basename + '_firings.mda'
        firings_out = file

        metrics_out_fname = mda_basename + '_metrics.json'

        # check if these outputs have already been created, skip if they have
        existing_files = 0
        output_files = [firings_out,
                        # filt_out_fname,
                        # pre_out_fname,
                        metrics_out_fname,
                        # masked_out_fname,
                        ]
        for outfile in output_files:
            if os.path.exists(outfile):
                existing_files += 1

        if existing_files != len(output_files):
            # then the file has not already been sorted, return True
            return True

    # already checked if position file exists so we don't need to do that

    # check if tetrodes/cut files have been converted already
    # filt_fnames = [os.path.join(directory, file) for file in os.listdir(
    #     directory) if '_filt.mda' in file if os.path.basename(tint_basename) in file]

    # for filt_filename in filt_fnames:
    for file in firing_fnames:
        # mda_basename = os.path.splitext(filt_filename)[0]
        mda_basename = os.path.splitext(file)[0]
        mda_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

        tetrode = int(mda_basename[find_sub(mda_basename, '_')[-1] + 2:])
        tetrode_filepath = '%s.%d' % (os.path.join(directory, output_basename), tetrode)

        if not os.path.exists(tetrode_filepath):
            # the tetrode has not been created yet, return True
            return True

        cut_filename = '%s_%d.cut' % (os.path.join(directory, output_basename), tetrode)

        if not os.path.exists(cut_filename):
            # the cut file has not been created yet, return True
            return True

            # check if set file has been converted
    set_filename = '%s.set' % os.path.join(directory, output_basename)
    if not os.path.exists(set_filename):
        return True

    # get eeg and egf files + channels to convert
    eeg_filenames, egf_filenames, eeg_channels = get_eeg_channels(probe_map, directory, output_basename,
                                                                  channels=convert_channels)
    # check if the .eeg/.egf files have been created yet
    for file in eeg_filenames:
        if not os.path.exists(file):
            # the eeg file does not exist
            return True

    for file in egf_filenames:
        if not os.path.exists(file):
            # the egf file does not exist
            return True

    # if you made it this far, then everything has been converted, return False
    return False


def cleanup_files(directory, tint_basename, delete_pre=True, delete_firings=False, delete_masked=True,
                  delete_filt=True, delete_raw=True):
    """
    This function will iterate through files that were created, and delete files that we don't need just to save space.
    :return:
    """
    delete_files = []

    if delete_pre:
        pre_filenames = [os.path.join(directory, file) for file in os.listdir(
            directory) if '_pre.mda' in file if os.path.basename(tint_basename) in file]
        delete_files.extend(pre_filenames)

    if delete_firings:
        firing_filenames = [os.path.join(directory, file) for file in os.listdir(
            directory) if '_firings.mda' in file if os.path.basename(tint_basename) in file]
        delete_files.extend(firing_filenames)

    if delete_masked:
        masked_filenames = [os.path.join(directory, file) for file in os.listdir(
            directory) if '_masked.mda' in file if os.path.basename(tint_basename) in file]
        delete_files.extend(masked_filenames)

    if delete_filt:
        filt_filenames = [os.path.join(directory, file) for file in os.listdir(
            directory) if '_filt.mda' in file if os.path.basename(tint_basename) in file]
        delete_files.extend(filt_filenames)

    if delete_raw:
        filt_filenames = [os.path.join(directory, file) for file in os.listdir(
            directory) if '_raw.mda' in file if os.path.basename(tint_basename) in file]
        delete_files.extend(filt_filenames)

    if len(delete_files) > 0:
        for file in delete_files:
            os.remove(file)


def get_start_stop(session_files, directory, tint_basename, self=None):
    """
    This function will look through the files and determine where the start and stop time is, according
    to the digital and analog signals it reads.
    """
    # check if there are digital inputs in the intan system that will be used to sync behavior w/ ephys data

    file_header = read_header(session_files[0])  # read the file header information from a session file
    recording_Fs = int(file_header['sample_rate'])

    if file_header['num_board_dig_in_channels'] > 0 or file_header['num_board_adc_channels'] > 0:
        # then we have the analog/digitaldigital values

        # reading the analog/digital values #
        data_digital_in = np.array([])
        data_analog_in = np.array([])
        for session_file in sorted(session_files, reverse=False):
            # Loads each session and appends them to create one matrix of data for the current tetrode
            # file_data = load_rhd.read_data(session_file)  # loading the .rhd data
            file_data = read_data(session_file, include_digital=True, include_analog=True, include_ephys=False)

            # Acquiring session information
            if file_header['num_board_dig_in_channels'] > 0:
                if data_digital_in.shape[0] == 0:
                    data_digital_in = file_data['board_dig_in_data']
                else:
                    data_digital_in = np.concatenate((data_digital_in, file_data['board_dig_in_data']), axis=1)

            if file_header['num_board_adc_channels'] > 0:
                if data_analog_in.shape[0] == 0:
                    data_analog_in = file_data['board_adc_data']
                else:
                    data_analog_in = np.concatenate((data_analog_in, file_data['board_adc_data']), axis=1)

        start_index, stop_index = get_data_limits(directory, tint_basename, data_digital_in, data_analog_in,
                                                           self=self)

        duration = np.floor((stop_index - start_index) / recording_Fs)
        # we floor the value so we can have a integer value for the duration
        n = int(duration * recording_Fs)

        # creating a new stop_index with the new number of samples that will
        # allow for the rounded integer value duration
        stop_index = start_index + n

        return start_index, stop_index

    return None, None


def convert_intan_mountainsort(session_files, interpolation=True, whiten='true',
                               detect_interval=10,
                               detect_sign=0, detect_threshold=3, freq_min=300, freq_max=6000,
                               mask_threshold=6,
                               flip_sign=False, software_rereference=True, reref_method='sd',
                               reref_channels=None, masked_chunk_size=None, mask=True,
                               mask_num_write_chunks=100,
                               clip_size=50, notch_filter=False, desired_Fs=48e3,
                               positionSampleFreq=50,
                               pre_spike_samples=10, post_spike_samples=40, rejthreshtail=43,
                               rejstart=30,
                               rejthreshupper=100, rejthreshlower=-100,
                               remove_spike_percentage=1,
                               clip_scalar=1,
                               clip_method='max',
                               num_features=10,
                               max_num_clips_for_pca=1000,
                               remove_outliers=False, eeg_channels='first', self=None):
    """
    This function will convert the current sessions data to .mda format, MountainSort the data,
    then convert the data to Tint (creating .egf/.eeg, .pos, .N, and .cut files.)

    :param session_files:
    :param interpolation:
    :param whiten:
    :param detect_interval:
    :param detect_sign:
    :param detect_threshold:
    :param freq_min:
    :param freq_max:
    :param mask_threshold:
    :param flip_sign:
    :param software_rereference:
    :param reref_method:
    :param reref_channels:
    :param masked_chunk_size:
    :param mask:
    :param mask_num_write_chunks:
    :param clip_size:
    :param notch_filter:
    :param desired_Fs:
    :param positionSampleFreq:
    :param pre_spike_samples:
    :param post_spike_samples:
    :param rejthreshtail:
    :param rejstart:
    :param rejthreshupper:
    :param rejthreshlower:
    :param remove_spike_percentage:
    :param clip_scalar:
    :param clip_method:
    :param num_features:
    :param max_num_clips_for_pca:
    :param remove_outliers:
    :param eeg_channels:
    :param self:
    :return:
    """

    directory = os.path.dirname(session_files[0])

    tint_basename = os.path.basename(os.path.splitext(sorted(session_files, reverse=False)[0])[0])
    tint_fullpath = os.path.join(directory, tint_basename)

    output_basename = '%s_ms' % tint_fullpath

    # pos_filename = output_basename + '.pos'
    pos_filename = tint_fullpath + '.pos'

    set_filename = output_basename + '.set'

    if interpolation:
        # the data will be interpolated to fit the desired Fs
        Fs = int(desired_Fs)
    else:
        # else just use whatever sample rate the data was recorded at
        Fs = int(read_header(session_files[0])['sample_rate'])

    # implement software re-referencing if the user wants to
    # check if there's a reference channel chosen
    file_header = read_header(session_files[0])

    # get the probe value from the notes
    probe = get_probe_name(session_files[0])

    probe_map = tetrode_map[probe]

    data_start_index, data_stop_index = get_start_stop(session_files, directory, tint_basename, self=self)

    if 'reference_channel' in file_header.keys():
        # Intan USB does not have this option
        reference_channel = file_header['reference_channel']
        if reference_channel != 'n/a':
            reference_channel = get_ref_index(file_header['amplifier_channels'],
                                              reference_channel)

            # TODO: add it so that if you want to change software references

            msg = '[%s %s]: Reference channel chosen during session.' % \
                  (str(datetime.datetime.now().date()),
                   str(datetime.datetime.now().time())[:8])
            if self:
                self.LogAppend.myGUI_signal_str.emit(msg)
            else:
                print(msg)

            reref_channels = [reference_channel]
            reref_method = None

    reref_data = None
    # getting re-referencing data
    if software_rereference:
        msg = '[%s %s]: Finding re-reference data!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8])
        if self:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        if reref_channels is None:
            if reref_method == 'sd':
                reref_data, reref_channels = get_reref_data(session_files, probe_map,
                                                            mode=reref_method,
                                                            start_index=data_start_index,
                                                            stop_index=data_stop_index)
            elif reref_method == 'avg':
                reref_data = get_reref_data(session_files, probe_map,
                                            mode=reref_method,
                                            start_index=data_start_index,
                                            stop_index=data_stop_index)
        else:
            reref_method = None  # we gave the channels, so set to None
            reref_data, reref_channels = get_reref_data(session_files,
                                                        probe_map,
                                                        channel=reref_channels,
                                                        start_index=data_start_index,
                                                        stop_index=data_stop_index)
    else:
        # deciding not to software re-reference
        reref_channels = None
        reref_data = None
        reref_method = None

    # convert the intan data to .mda so they can be analyzed
    sort_duration = intan2mda(session_files, interpolation=interpolation,
                              notch_filter=notch_filter,
                              flip_sign=flip_sign,
                              software_rereference=software_rereference,
                              reref_data=reref_data,
                              reref_channels=reref_channels,
                              reref_method=reref_method,
                              desired_Fs=desired_Fs,
                              start_index=data_start_index,
                              stop_index=data_stop_index,
                              self=self)

    # sort the mda data
    sort_intan(directory, tint_fullpath, Fs,
               whiten=whiten,
               detect_interval=detect_interval,
               detect_sign=detect_sign,
               detect_threshold=detect_threshold,
               freq_min=freq_min,
               freq_max=freq_max,
               mask_threshold=int(mask_threshold),
               mask=mask,
               masked_chunk_size=masked_chunk_size,
               mask_num_write_chunks=mask_num_write_chunks,
               clip_size=clip_size,
               num_features=num_features,
               max_num_clips_for_pca=max_num_clips_for_pca,
               self=self)

    # create positions
    convert_position(session_files, pos_filename, positionSampleFreq, output_basename, sort_duration, self=self)

    # at this point we can create the tetrode and cut file (without the tetrode header), we need some extra
    # parameters for the header so I will do this first to create the clipping (and thus the gain) values that will
    # be put into the set file. Create the set file, and then add the headers using those parameters
    batch_basename_tetrodes(directory, tint_basename, output_basename, Fs,
                            pre_spike_samples=pre_spike_samples,
                            post_spike_samples=post_spike_samples,
                            detect_sign=detect_sign,
                            remove_spike_percentage=remove_spike_percentage,
                            remove_outliers=remove_outliers,
                            clip_method=clip_method,
                            mask=mask,
                            clip_scalar=clip_scalar,
                            self=self)

    # create the set file
    set_converted = convert_setfile(session_files, tint_basename, set_filename, Fs,
                                    pre_spike_samples=pre_spike_samples,
                                    post_spike_samples=post_spike_samples,
                                    rejthreshtail=rejthreshtail,
                                    rejstart=rejstart,
                                    rejthreshupper=rejthreshupper,
                                    rejthreshlower=rejthreshlower,
                                    self=self)

    # overwrite the tetrode files to add the headers
    batch_add_tetrode_headers(directory, tint_fullpath,
                              self=self)

    # create eeg / egf
    eeg_converted = convert_eeg(session_files, tint_basename, output_basename, Fs,
                                convert_channels=eeg_channels,
                                self=self)

    if set_converted or eeg_converted:
        # then we will overwrite the eeg filename parameters that were set since it is not taken care of in the
        # initial convert_setfile() function. We will only do this if a .set file or .eeg file is newly created
        overwrite_eeg_set_params(tint_fullpath, set_filename)

    # clean up any files to save space
    if mask:
        delete_masked = False
        delete_filt = True
    else:
        delete_filt = False
        delete_masked = True

    # we will always delete the preprocessed data, we have the output from the sorting so we won't really need it
    # we will save the firings in case we want to view the data in MountainView, and if the user decides to mask
    # the data we will delete the filt and keep the mask. If the user decides not to mask, we will keep the filtered
    # and delete the masked which won't exist anyways.

    msg = '[%s %s]: Deleting unnecessary intermediate files from MountainSort.' % \
          (str(datetime.datetime.now().date()),
           str(datetime.datetime.now().time())[:8])
    if self:
        self.LogAppend.myGUI_signal_str.emit(msg)
    else:
        print(msg)

    cleanup_files(directory, tint_basename, delete_pre=True, delete_firings=False, delete_masked=delete_masked,
                  delete_filt=delete_filt, delete_raw=True)

