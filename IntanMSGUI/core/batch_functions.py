import os, datetime, time
from PyQt5 import QtWidgets
from core.intan_mountainsort import convert_intan_mountainsort, validate_session
import json
from core.utils import find_sub
from core.rhd_utils import tetrode_map, tintRef2intan
from core.intan_rhd_functions import find_basename_files, get_probe_name
from core.default_parameters import masked_chunk_size, eeg_channels, mask_num_write_chunks, clip_size, rejthreshtail, \
    rejstart, rejthreshupper, rejthreshlower, positionSampleFreq, axona_refs, clip_scalar, reref_channels
import numpy as np

# TODO: Add Parameters officially


def BatchAnalyze(main_window, settings_window, directory):
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

                        whiten = settings_window.whiten_cb.isChecked()
                        if whiten:
                            whiten = 'true'
                        else:
                            whiten = 'false'

                        detect_sign = settings_window.detect_sign

                        detect_threshold = float(settings_window.detect_threshold_widget.text())
                        detect_interval = int(settings_window.detect_interval_widget.text())

                        main_window.LogAppend.myGUI_signal_str.emit(
                            '[%s %s]: Analyzing the following basename: %s!' % (
                                str(datetime.datetime.now().date()),
                                str(datetime.datetime.now().time())[
                                :8], rhd_session_file))

                        interpolation = settings_window.interpolate_cb.isChecked()
                        freq_min = float(settings_window.lower_cutoff.text())
                        freq_max = float(settings_window.upper_cutoff.text())
                        mask_threshold = float(settings_window.mask_threshold.text())
                        flip_sign = settings_window.flip_sign.isChecked()
                        software_rereference = settings_window.software_rereference.isChecked()
                        reref_method = settings_window.reref_method_combo.currentText()
                        mask = settings_window.mask.isChecked()

                        notch_filter = settings_window.notch_filter.isChecked()

                        desired_Fs = float(settings_window.interpolate_Fs.text())

                        pre_spike_samples = settings_window.pre_threshold_widget.value()
                        post_spike_samples = settings_window.post_threshold_widget.value()

                        if pre_spike_samples + post_spike_samples != clip_size:
                            if pre_spike_samples + post_spike_samples != clip_size:
                                raise ValueError(
                                    "the pre (%d) and post (%d) spike samples need to add up to the clip_size: %d." % (
                                    pre_spike_samples,
                                    post_spike_samples,
                                    int(clip_size)))

                        remove_spike_percentage = float(settings_window.remove_outliers_percentage.text())

                        remove_outliers = settings_window.remove_outliers.isChecked()

                        remove_method = settings_window.remove_method.currentText()

                        num_features = int(settings_window.num_features.text())
                        max_num_clips_for_pca = int(settings_window.max_num_clips_for_pca.text())

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
                                                   num_features=num_features,
                                                   max_num_clips_for_pca=max_num_clips_for_pca,
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
