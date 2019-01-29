import os
import numpy as np

import datetime

from core.tetrode_conversion import batch_basename_tetrodes, batch_add_tetrode_headers
from core.convert_position import convert_position
from core.eeg_conversion import convert_eeg, get_eeg_channels
from core.utils import find_sub
from core.intan2mda import intan2mda, get_reref_data
from core.mdaSort import sort_intan
from core.set_conversion import convert_setfile
from core.rhd_utils import tetrode_map

from core.intan_rhd_functions import read_header, get_probe_name, get_ref_index


def validate_session(rhd_basename_file, output_basename, convert_channels, self=None, verbose=True):
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

        mda_filename = '%s_T%d_raw.mda' % (os.path.join(directory, tint_basename), tetrode)

        if os.path.exists(mda_filename):
            converted_files += 1

        n_tetrodes += 1

    if n_tetrodes != converted_files:
        return True

    # check that all the tetrodes have been converted

    raw_fnames = [os.path.join(directory, file) for file in os.listdir(
        directory) if '_raw.mda' in file if tint_basename in file]

    eeg_filenames = []
    egf_filenames = []

    for file in raw_fnames:
        mda_basename = os.path.splitext(file)[0]
        mda_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

        masked_out_fname = mda_basename + '_masked.mda'
        firings_out = mda_basename + '_firings.mda'
        filt_out_fname = mda_basename + '_filt.mda'
        pre_out_fname = mda_basename + '_pre.mda'
        metrics_out_fname = mda_basename + '_metrics.json'

        # check if these outputs have already been created, skip if they have
        existing_files = 0
        output_files = [masked_out_fname, filt_out_fname, firings_out,
                        pre_out_fname, metrics_out_fname]
        for outfile in output_files:
            if os.path.exists(outfile):
                existing_files += 1

        if existing_files != len(output_files):
            # then the file has not already been sorted, return True
            return True

    # already checked if position file exists so we don't need to do that

    # check if tetrodes/cut files have been converted already
    filt_fnames = [os.path.join(directory, file) for file in os.listdir(
        directory) if '_filt.mda' in file if os.path.basename(tint_basename) in file]

    for filt_filename in filt_fnames:
        mda_basename = os.path.splitext(filt_filename)[0]
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


def convert_intan_mountainsort(session_files, interpolation=True, whiten='true',
                               detect_interval=10,
                               detect_sign=0, detect_threshold=3, freq_min=300, freq_max=6000,
                               mask_threshold=6,
                               flip_sign=False, software_rereference=True, reref_method='sd',
                               reref_channels=None, masked_chunk_size=None,
                               mask_num_write_chunks=100,
                               clip_size=50, notch_filter=False, desired_Fs=48e3,
                               positionSampleFreq=50,
                               pre_spike_samples=10, post_spike_samples=40, rejthreshtail=43,
                               rejstart=30,
                               rejthreshupper=100, rejthreshlower=-100,
                               remove_spike_percentage=1,
                               clip_scalar=1,
                               clip_method='max',
                               remove_outliers=False, eeg_channels='first', self=None):

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

    if 'reference_channel' in file_header.keys():
        # Intan USB does not have this option
        reference_channel = file_header['reference_channel']
        if reference_channel != 'n/a':
            reference_channel = get_ref_index(file_header['amplifier_channels'],
                                              reference_channel)
            # reference_channel = int(reference_channel.split('-')[-1]) + 1

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
                                                           mode=reref_method)
            elif reref_method == 'avg':
                reref_data = get_reref_data(session_files, probe_map,
                                            mode=reref_method)
        else:
            reref_method = None  # we gave the channels, so set to None
            reref_data, reref_channels = get_reref_data(session_files,
                                                       probe_map,
                                                       channel=reref_channels)
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
                              self=self)

    # sort the mda data
    sort_intan(directory, tint_fullpath, Fs, whiten=whiten, detect_interval=detect_interval,
               detect_sign=detect_sign,
               detect_threshold=detect_threshold, freq_min=freq_min, freq_max=freq_max,
               mask_threshold=mask_threshold,
               masked_chunk_size=masked_chunk_size, mask_num_write_chunks=mask_num_write_chunks,
               clip_size=clip_size, self=self)

    # create positions
    convert_position(session_files, pos_filename, positionSampleFreq, output_basename, sort_duration, self=self)

    # at this point we can create the tetrode and cut file (without the tetrode header), we need some extra
    # parameters for the header so I will do this first to create the clipping (and thus the gain) values that will
    # be put into the set file. Create the set file, and then add the headers using those parameters
    batch_basename_tetrodes(directory, tint_basename, output_basename, Fs,
                            pre_spike_samples=pre_spike_samples,
                            post_spike_samples=post_spike_samples, detect_sign=detect_sign,
                            remove_spike_percentage=remove_spike_percentage,
                            remove_outliers=remove_outliers, clip_method=clip_method,
                            clip_scalar=clip_scalar,
                            self=self)

    # create the set file
    convert_setfile(session_files, tint_basename, set_filename, Fs,
                    pre_spike_samples=pre_spike_samples,
                    post_spike_samples=post_spike_samples, rejthreshtail=rejthreshtail,
                    rejstart=rejstart,
                    rejthreshupper=rejthreshupper, rejthreshlower=rejthreshlower, self=self)

    # overwrite the tetrode files to add the headers
    batch_add_tetrode_headers(directory, tint_fullpath, self=self)

    # create eeg / egf
    convert_eeg(session_files, tint_basename, output_basename, Fs,
                convert_channels=eeg_channels, self=self)
