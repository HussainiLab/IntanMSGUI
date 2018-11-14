import struct
import numpy as np
import os


def write_tetrode(filepath, spike_times, spike_data, tetrode_parameters):
    # Note, this can be optimized instead of writing each tetrode one at a time
    session_path, session_filename = os.path.split(filepath)

    with open(filepath, 'w') as f:
        date = 'trial_date %s' % (tetrode_parameters['trial_date'])
        time_head = '\ntrial_time %s' % (tetrode_parameters['trial_time'])
        experimenter = '\nexperimenter %s' % (tetrode_parameters['experimenter'])
        comments = '\ncomments %s' % (tetrode_parameters['comments'])

        duration = '\nduration %d' % (int(tetrode_parameters['duration']))

        sw_vers = '\nsw_version %s' % (tetrode_parameters['sw_version'])
        num_chans = '\nnum_chans 4'
        timebase_head = '\ntimebase %d hz' % (96000)
        bp_timestamp = '\nbytes_per_timestamp %d' % (4)
        # samps_per_spike = '\nsamples_per_spike %d' % (int(Fs*1e-3))
        samps_per_spike = '\nsamples_per_spike %d' % (int(tetrode_parameters['samples_per_spike']))
        sample_rate = '\nsample_rate %d hz' % (int(tetrode_parameters['rawRate']))
        b_p_sample = '\nbytes_per_sample %d' % (1)
        # b_p_sample = '\nbytes_per_sample %d' % (4)
        spike_form = '\nspike_format t,ch1,t,ch2,t,ch3,t,ch4'

        num_spikes = '\nnum_spikes %d' % (spike_data.shape[1])
        start = '\ndata_start'

        write_order = [date, time_head, experimenter, comments, duration, sw_vers, num_chans, timebase_head,
                       bp_timestamp,
                       samps_per_spike, sample_rate, b_p_sample, spike_form, num_spikes, start]

        f.writelines(write_order)

    with open(filepath, 'rb+') as f:
        for i in np.arange(spike_data.shape[1]):
            data = spike_data[:, i, :]
            spike_t = spike_times[i]
            write_list = []
            for i in np.arange(data.shape[0]):
                write_list.append(struct.pack('>i', int(spike_t)))
                write_list.append(struct.pack('<%db' % (tetrode_parameters['samples_per_spike']),
                                              *[int(sample) for sample in data[i, :].tolist()]))

            f.seek(0, 2)
            f.writelines(write_list)

    with open(filepath, 'rb+') as f:
        f.seek(0, 2)
        f.write(bytes('\r\ndata_end\r\n', 'utf-8'))


def get_cut_values(spike_times, indices, cell_number, desired_events_indices):
    # initializing the cut value that will be output
    cut = np.zeros(len(desired_events_indices))

    # produce a dictionary where the key is the chunk index, and the value is a list of events within each chunk
    chunk_events_values = get_chunk_events(spike_times, indices[:, 1])

    event_values, chunk_indices = process_event_values(spike_times, chunk_events_values, desired_events_indices, indices)

    cut_bool = np.in1d(spike_times, event_values)

    if not np.array_equal(spike_times[cut_bool], event_values):
        raise CutError("Error, the sliced spike_times should match the event_values")

    cut[chunk_indices] = cell_number[cut_bool]

    return cut.astype(int)


def create_cut(set_filename, tags_bool, merged=False, pre_threshold=10, post_threshold=40, version='ms3', whiten=True,
               self=None):
    """
        This will take the sorted output and create a .cut file that Tint can understand.

        Example:
            set_filename = 'C:\\example\\example_session.set'
            create_cut(set_filename)

        Args:
            set_filename (str): the fullpath for the .set file you want to analyze.
            curated (bool): True means you want to curate the sorted data, False means you do not. Curation will further
                sort the data by combining similar cells, etc. Note that if you curate on data with a lot of noise you
                could end up with no cells.
            pre_threshold (int): the number of samples before threshold that you want to save for the waveform.
            version (string): {'js', 'ms3'}, 'js' for the new javascript version, 'ms3' for the old MountainSort3.
            self (object): self is the main window of a GUI that has a LogAppend method that will append this message to
                a log (QTextEdit), and a custom signal named myGUI_signal_str that would was hooked up to LogAppend to
                append messages to the log.

        Returns:
            None
    """

    basename = os.path.basename(os.path.splitext(set_filename)[0])
    directory, _ = os.path.split(set_filename)
    tetrodes = get_active_tetrode(set_filename)
    tetrode_files = [os.path.join(directory, '%s.%d' % (basename, tetrode)) for tetrode in tetrodes]
    # directory, tetrode_files = find_tet(set_filename)  # find the tetrode files

    # just in case the curated version doesn't work and it runs a non-curated sort, we want to convert both to .cut
    if not merged:
        if whiten:
            firings_mda = get_mda_files(set_filename, 'firings_%s' % version)
        else:
            firings_mda = get_mda_files(set_filename, 'firings_nowhiten_%s' % version)
    else:
        if whiten:
            firings_mda = get_mda_files(set_filename, 'firingsmerged_%s' % version)
        else:
            firings_mda = get_mda_files(set_filename, 'firingsmerged_nowhiten_%s' % version)

    for spike_filename in firings_mda:

        msg = '[%s %s]: Reading the following file to obtain event data: %s!' % (
            str(datetime.datetime.now().date()),
            str(datetime.datetime.now().time())[
            :8], spike_filename)

        if self:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)

        if whiten:
            tetrode = get_rawmda_tetrode(spike_filename, unders=3)
        else:
            tetrode = get_rawmda_tetrode(spike_filename, unders=4)

        tetrode_fname = [file for file in tetrode_files if os.path.splitext(file)[1] == ('.%d' % tetrode)][0]

        if not merged:
            if whiten:
                cut_filename = os.path.splitext(tetrode_fname)[0] + '_%d_%s.cut' % (tetrode, version)
            else:
                cut_filename = os.path.splitext(tetrode_fname)[0] + '_nowhiten_%d_%s.cut' % (tetrode, version)
        else:
            if whiten:
                cut_filename = os.path.splitext(tetrode_fname)[0] + '_merged_%d_%s.cut' % (tetrode, version)
            else:
                cut_filename = os.path.splitext(tetrode_fname)[0] + '_merged_nowhiten_%d_%s.cut' % (tetrode, version)

        if os.path.exists(cut_filename):
            if tags_bool:
                # tags bool being true is required, that means that all previous analysis has been done before so it is
                # likely that this cut file already has the correct values.
                msg = '[%s %s]: The following .cut file already exists: %s, skipping!' % (
                    str(datetime.datetime.now().date()),
                    str(datetime.datetime.now().time())[
                    :8], cut_filename)

                if self:
                    self.LogAppend.myGUI_signal_str.emit(msg)
                else:
                    print(msg)
                continue

        A, code = readMDA(spike_filename)

        spike_times = (A[1, :]).astype(int)  # at this stage it is in index values (0-based)

        # remove any repeat spike times
        repeat_spikes = np.where(np.diff(spike_times) == 0)[0]

        A = np.delete(A, repeat_spikes + 1, axis=1)

        # spike_channel = A[0, :].astype(int)  # the channel which the spike belongs to
        spike_times = (A[1, :]).astype(int)  # at this stage it is in index values (0-based)
        cell_number = A[2, :].astype(int)

        ts, ch1, ch2, ch3, ch4, spikeparam = getspikes(tetrode_fname)  # sec, bits, bits, bits, bits, dict

        snippets_concat = np.vstack((ch1, ch2, ch3, ch4)).reshape((4, -1))

        ch1 = None
        ch2 = None
        ch3 = None
        ch4 = None

        Fs = spikeparam['sample_rate']

        ts_indices = np.rint(np.multiply(ts.flatten(), Fs)).astype(int)  # get the index value of threshold cross
        ts_indices = ts_indices - pre_threshold + 1  # ensures that index value represents the start of the chunk

        data_snip_indices = get_snip_indices(snippets_concat, ts_indices)

        start_i = data_snip_indices[::50].reshape((-1, 1))
        stop_i = data_snip_indices[49::50].reshape((-1, 1)) + 1  # add 1 so it includes the stop value

        indices = np.hstack((np.arange(len(start_i)).reshape((-1, 1)), start_i, stop_i))

        # we desire to get events that are closest to the middle of the chunk since mountainsort will take equal
        # portions to the start and end of the event (unlike Tint which will do 10 samples pre, and 40 post threshold)

        desired_events_indices = indices[:, 1] + int((pre_threshold + post_threshold)/2)

        cut = get_cut_values(spike_times, indices, cell_number, desired_events_indices)

        metric_files = get_metric_files(set_filename)

        if merged:
            if whiten:
                version_string = '_metricsmerged_%d_%s.json' % (tetrode, version)
            else:
                version_string = '_metricsmerged_nowhiten_%d_%s.json' % (tetrode, version)
        else:
            if whiten:
                version_string = '_metrics_%d_%s.json' % (tetrode, version)
            else:
                version_string = '_metrics_nowhiten_%d_%s.json' % (tetrode, version)

        metric_files = [file for file in metric_files if version_string in file]

        try:
            mua_cells = get_mua_cells(metric_files[0])
        except:
            x = 1
            pass

        cut_cont, cut_dict = get_tint_cut(cut, mua_cells)

        write_cut(cut_filename, cut_cont, basename=basename)

        msg = '[%s %s]: Completed writing the following .cut file: %s!' % (
            str(datetime.datetime.now().date()),
            str(datetime.datetime.now().time())[
            :8], cut_filename)

        if self:
            self.LogAppend.myGUI_signal_str.emit(msg)
        else:
            print(msg)