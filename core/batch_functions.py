import os, datetime, time
from PyQt5 import QtWidgets
from core.intan_mountainsort import convert_intan_mountainsort, validate_session
import json
from core.utils import find_sub
from core.rhd_utils import tetrode_map
from core.intan_rhd_functions import find_basename_files, get_probe_name
import numpy as np

# TODO: Add Parameters officially

# INTERPOLATE
interpolation = True  # if you want to interpolate, options: True or False, remember capital letter
desired_Fs = int(48e3)  # Sampling Frequency you will interpolate to

# WHITEN
whiten = 'true'  # if you want to whiten or not, 'true' or 'false', lower case letters
# whiten = 'false'

# THRESHOLD
flip_sign = True  # if you want to flip the signal (multiply by -1)

detect_interval = 50  # it will split up the data into segments of this value and find a peak/trough for each segment
detect_sign = 0  # 0 = positive and negative peaks, 1 = positive peaks, -1 = negative peaks

if whiten == 'true':
    # the threshold of the data, if whitened, data is normalized to standard deviations, so a value of 3
    detect_threshold = 3
    # would mean 3 standard deviations away from baseline. If not whitened treat it like a Tint threshold, i.e. X bits.
    # essentially bits vs standard deviations.
else:
    # you can grab a value from a set file with this animal, the parameter is called 'threshold', it is however in
    # 16bit, so divide the value by 256 to convert to 8bit, since the thresholding is in 8 bit.
    detect_threshold = 33

# BANDPASS
freq_min = 300  # min frequency to bandpass at
freq_max = 6000  # max freq to bandpass at

# EEG Settings

eeg_channels = 'first'  # this will save the 1st channel as an .eeg in each tetrode
# eeg_channels = 'all'  # this will save all the channels as their own .eeg file
# eeg_channels = [W, X, Y, Z]  # list of channels if you want to specify which ones to use

# MASK
# The amount of indices that you will segment for masking artifacts. if you leave as None it will
# mask = True
mask = False

masked_chunk_size = None
# use a value of Fs/10 or Fs/20, I forget

mask_num_write_chunks = 100  # this is how many of the chunk segments will be written at the same time, mainly just for
# optimizing write speeds, just try to leave it as ~100k - 300k samples.

mask_threshold = 6  # number of standard deviations the Root of the Sum of the Squares (RSS) of all the segments
# (of masked chunk size). If the RSS is above this threshold it will assume it is artifact, and set all values in this
# segment to zero.

# COMMON MODE PARAMETERS (essentially software re-referencing)
# software re-ref PARAMETERS (essentially software re-referencing)
software_rereference = True  # if you want to remove the common signal (mean values)
# between channels, set to True, otherwise False

# reref_method=None
reref_method = 'sd'  # methods of which to remove signal, 'sd' will caculate the standard deviation of every channel
# and choose the channels with the two lowest values and remove those signals from the rest (the 2nd lowest will be
# removed from the 1st lowest). Somewhat like what you guys do visually in Axona.

reref_channels = 'auto'  # below we have a dictionary of previously chosen values
# depending on the mouse
# if set to auto, it will choose those values.

# reref_channels = [16, 9]  # if you wanted to just say which
# channels to subtract from the rest, do it here,
# reref_channels = None
# it will override the automatic stuff. Essentially look through .set files

clip_size = 50  # samples per spike, default 50

# if you want to add a notch. However, it is already filtered using freq_min, so this doesn't really help
notch_filter = True
# unless of course your freqmin is below 60 Hz, default False

positionSampleFreq = 50  # sampling frequency of position, default 50

pre_spike_samples = 15  # number of samples pre-threshold samples to take, default 10
post_spike_samples = 35  # number of post threshold samples to take, default 40

if pre_spike_samples + post_spike_samples != clip_size:
    raise ValueError("the pre (%d) and post (%d) spike samples need to add up to the clip_size: %d." % (pre_spike_samples,
                                                                                                        post_spike_samples,
                                                                                                        int(clip_size)))

# Axona Artifact Rejection Criteria, I'd just leave these. They are in the manual
rejthreshtail = 43  # I think, the latter 20-30% can't be above this value ( I think)
rejstart = 30  #
rejthreshupper = 100  # if 1st sample is above this value in bits, it will discard
rejthreshlower = -100  # if 1st sample is below this value in bits, it will discard

# The percentage of spikes to remove as outliers, this will make it so they don't make the majority of the
# spikes look really small in amplitude
remove_outliers = True  # if False, it won't remove outliers, if True, it will.
remove_spike_percentage = 5  # percent value, default 1, haven't experimented much with this

remove_method = 'max'  # this will find the max of the peak values (or min if it's negative)
# and set that as the clipping value

clip_scalar = 1.05  # this will multiply the clipping value found via the remove_method method,
# and then scale by this value.

# miscellaneous
# self = None  # this is code jargin for object oriented programming, mainly used for GUI's, we don't need this
# just needs to be set in the function so I have it set to None.

axona_refs = {
    'b6_august_18_1': [4, 3],
    'b6_august_18_2': [4, 3],
    'j20_sleep_1': [4, 3],
    'j20_sleep_2': [4, 3],
    'b6_sep_18_1': [4, 3],
}

# END PARAMETERS


def tintRef2intan(tint_refs, tetrode_map, probe):
    """
    Given a list of tint_references (0-based index), it will return the channel
    (1-based indexing) that this tint channel represents in Intan.

    Tint is 0-based because that is what comes out of Tint (in the set file).
    Intan is 1-based because it's just easy to think about a 16 channel probe
    as having channels from 1-16.
    """
    probe_map = tetrode_map[probe]

    tetrode_channels = []
    for key in sorted(probe_map.keys()):
        tetrode_channels.extend(probe_map[key])

    tetrode_channels = np.asarray(tetrode_channels)

    return list(tetrode_channels[tint_refs])


def intanRef2tint(intan_refs, tetrode_map, probe):
    """
    Given a list of intan references (1-based index), it will return the tinta channel
    (0-based indexing) that this intan channel represents.

    Tint is 0-based because that is what comes out of Tint (in the set file).
    Intan is 1-based because it's just easy to think about a 16 channel probe
    as having channels from 1-16.
    """
    probe_map = tetrode_map[probe]

    tetrode_channels = []
    for key in sorted(probe_map.keys()):
        tetrode_channels.extend(probe_map[key])

    tetrode_channels = np.asarray(tetrode_channels)

    tint_refs = []
    for channel in intan_refs:
        tint_refs.append(np.where(tetrode_channels == channel)[0][0])

    return tint_refs


def BatchAnalyze(main_window, directory):
    # ------- making a function that runs the entire GUI ----------

    # checks if the settings are appropriate to run analysis
    # klusta_ready = check_klusta_ready(main_window, directory)

    # get settings

    directory = os.path.realpath(directory)

    batch_analysis_ready = True

    if batch_analysis_ready:

        main_window.LogAppend.myGUI_signal_str.emit(
            '[%s %s]: Analyzing the following directory: %s!' % (str(datetime.datetime.now().date()),
                                                                 str(datetime.datetime.now().time())[
                                                                 :8], directory))

        if not main_window.nonbatch:
            # message that shows how many files were found
            main_window.LogAppend.myGUI_signal_str.emit(
                '[%s %s]: Found %d sub-directories in the directory!#Red' % (str(datetime.datetime.now().date()),
                                                               str(datetime.datetime.now().time())[
                                                               :8], main_window.directory_queue.topLevelItemCount()))

        else:
            directory = os.path.dirname(directory)

        if main_window.directory_queue.topLevelItemCount() == 0:
            # main_window.AnalyzeThread.quit()
            # main_window.AddSessionsThread.quit()
            if main_window.nonbatch:
                main_window.choice = ''
                main_window.LogError.myGUI_signal_str.emit('InvDirNonBatch')
                while main_window.choice == '':
                    time.sleep(0.2)
                main_window.stopBatch()
                return
            else:
                main_window.choice = ''
                main_window.LogError.myGUI_signal_str.emit('InvDirBatch')
                while main_window.choice == '':
                    time.sleep(0.2)

                if main_window.choice == QtWidgets.QMessageBox.Abort:
                    main_window.stopBatch()
                    return

        # save directory settings
        with open(main_window.directory_settings, 'w') as filename:
            if not main_window.nonbatch:
                save_directory = directory
            else:
                if main_window.directory_queue.topLevelItemCount() > 0:
                    sub_dir = main_window.directory_queue.topLevelItem(0).data(0, 0)
                    save_directory = os.path.join(directory, sub_dir)
                else:
                    save_directory = directory

            settings = {"directory": save_directory}
            json.dump(settings, filename)

        # ----------- cycle through each file  ------------------------------------------

        while main_window.directory_queue.topLevelItemCount() > 0:

            main_window.directory_item = main_window.directory_queue.topLevelItem(0)

            if not main_window.directory_item:
                continue
            else:
                main_window.current_subdirectory = main_window.directory_item.data(0, 0)

                # check if the directory exists, if not, remove it

                if not os.path.exists(os.path.join(directory, main_window.current_subdirectory)):
                    main_window.top_level_taken = False
                    main_window.RemoveQueueItem.myGUI_signal_str.emit(str(0))
                    while not main_window.top_level_taken:
                        time.sleep(0.1)
                    continue

            while main_window.directory_item.childCount() != 0:

                main_window.current_session = main_window.directory_item.child(0).data(0, 0)
                main_window.child_data_taken = False
                main_window.RemoveSessionData.myGUI_signal_str.emit(str(0))
                while not main_window.child_data_taken:
                    time.sleep(0.1)

                sub_directory = main_window.directory_item.data(0, 0)

                directory_ready = False

                main_window.LogAppend.myGUI_signal_str.emit(
                    '[%s %s]: Checking if the following directory is ready to analyze: %s!' % (
                        str(datetime.datetime.now().date()),
                        str(datetime.datetime.now().time())[
                        :8], str(sub_directory)))

                if main_window.directory_item.childCount() == 0:
                    main_window.top_level_taken = False
                    main_window.RemoveQueueItem.myGUI_signal_str.emit(str(0))
                    while not main_window.top_level_taken:
                        time.sleep(0.1)
                try:

                    if not os.path.exists(os.path.join(directory, sub_directory)):
                        main_window.top_level_taken = False
                        main_window.RemoveQueueItem.myGUI_signal_str.emit(str(0))
                        while not main_window.top_level_taken:
                            time.sleep(0.1)
                        continue
                    else:

                        analysis_directory = os.path.join(directory, main_window.current_subdirectory)

                        rhd_session_file = main_window.current_session
                        rhd_basename = rhd_session_file[:find_sub(rhd_session_file, '_')[-2]]

                        session_files = find_basename_files(rhd_basename, os.path.join(directory, sub_directory))

                        rhd_session_fullfile = os.path.join(directory, sub_directory, rhd_session_file + '.rhd')

                        tint_basename = os.path.basename(os.path.splitext(rhd_session_fullfile)[0])
                        tint_fullpath = os.path.join(analysis_directory, tint_basename)
                        output_basename = '%s_ms' % tint_fullpath
                        session_valid = validate_session(rhd_session_fullfile, output_basename, eeg_channels,
                                                         self=main_window, verbose=False)

                        if not session_valid:
                            message = '[%s %s]: The following session has already been analyzed: %s' % (
                                str(datetime.datetime.now().date()),
                                str(datetime.datetime.now().time())[
                                :8], os.path.basename(
                                    tint_fullpath))

                            main_window.LogAppend.myGUI_signal_str.emit(message)
                            continue

                        # find the session with our rhd file in it
                        session_files = [sub_list for sub_list in session_files if rhd_session_fullfile in sub_list][0]

                        if type(session_files) != list:
                            # if there is only one file in the list, the output will not be a list
                            session_files = [session_files]

                        probe = get_probe_name(session_files[0])

                        if not main_window.nonbatch:
                            mouse = os.path.basename(os.path.dirname(analysis_directory))
                        else:
                            mouse = os.path.basename(analysis_directory)

                        global reref_channels, axona_refs

                        if mouse not in axona_refs.keys():
                            for key in axona_refs.keys():
                                if key in session_files[0]:
                                    mouse = key

                        if reref_channels == 'auto':
                            reref_channels = tintRef2intan(axona_refs[mouse],
                                                           tetrode_map,
                                                           probe)
                            print('The following reref_channels were chosen: ', reref_channels)

                        whiten = main_window.whiten_cb.isChecked()
                        if whiten:
                            whiten = 'true'
                        else:
                            whiten = 'false'

                        detect_sign = main_window.detect_sign

                        detect_threshold = main_window.detect_threshold_widget.value()
                        # version = main_window.version
                        detect_interval = main_window.detect_interval_widget.value()

                        main_window.LogAppend.myGUI_signal_str.emit(
                            '[%s %s]: Analyzing the following basename: %s!' % (
                                str(datetime.datetime.now().date()),
                                str(datetime.datetime.now().time())[
                                :8], rhd_session_file))

                        convert_intan_mountainsort(session_files, interpolation=interpolation, whiten=whiten,
                                                   detect_interval=detect_interval,
                                                   detect_sign=detect_sign, detect_threshold=detect_threshold,
                                                   freq_min=freq_min,
                                                   freq_max=freq_max, mask_threshold=mask_threshold,
                                                   flip_sign=flip_sign,
                                                   software_rereference=software_rereference,
                                                   reref_method=reref_method,
                                                   reref_channels=reref_channels,
                                                   mask=mask,
                                                   masked_chunk_size=masked_chunk_size,
                                                   mask_num_write_chunks=mask_num_write_chunks,
                                                   clip_size=clip_size,
                                                   notch_filter=notch_filter,
                                                   desired_Fs=desired_Fs,
                                                   positionSampleFreq=positionSampleFreq,
                                                   pre_spike_samples=pre_spike_samples,
                                                   post_spike_samples=post_spike_samples,
                                                   rejthreshtail=rejthreshtail, rejstart=rejstart,
                                                   rejthreshupper=rejthreshupper, rejthreshlower=rejthreshlower,
                                                   remove_spike_percentage=remove_spike_percentage,
                                                   remove_outliers=remove_outliers,
                                                   clip_scalar=clip_scalar,
                                                   clip_method=remove_method,
                                                   eeg_channels=eeg_channels,
                                                   self=main_window)

                        main_window.analyzed_sessions.append(main_window.current_session)

                        main_window.LogAppend.myGUI_signal_str.emit(
                            '[%s %s]: Finished analyzing the following basename: %s!' % (
                                str(datetime.datetime.now().date()),
                                str(datetime.datetime.now().time())[
                                :8], rhd_session_file))

                except NotADirectoryError:
                    # if the file is not a directory it prints this message
                    main_window.LogAppend.myGUI_signal_str.emit(
                        '[%s %s]: %s is not a directory!' % (
                            str(datetime.datetime.now().date()),
                            str(datetime.datetime.now().time())[
                            :8], str(sub_directory)))
                    continue
