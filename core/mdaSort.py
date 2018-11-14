from core.wsl_terminal import BashConfigure
import numpy as np
import os
from core.utils import find_sub
import time
import datetime


def check_file_complete(filepath, delta_time=5):

    mtime = 0
    new_mtime = os.path.getmtime(filepath)

    while mtime != new_mtime:
        mtime = new_mtime
        time.sleep(delta_time)
        new_mtime = os.path.getmtime(filepath)

    return


def get_ubuntu_path(filepath):
    # get the drive letter

    drive_letter_i = filepath.find(':/')

    if drive_letter_i == -1:
        drive_letter_i = filepath.find(':\\')

    drive_letter = filepath[:drive_letter_i].lower()

    i = 1
    while drive_letter_i + i == '/':
        i += 1

    remaining_path = filepath[drive_letter_i + i + 1:]
    linux_path = '/mnt/%s/%s' % (drive_letter, remaining_path)

    # add single quotes so linux can understand the special characters
    if '(' in linux_path or ')' in linux_path:
        linux_path = "'%s'" % linux_path

    return os.path.normpath((linux_path)).replace('\\', '/')


def get_windows_filename(filename):
    # remove the single quotes if added
    if filename[0] == "'" and filename[-1] == "'":
        filename = filename[1:-1]

    filename_split = filename.split('/')
    mnt_i = np.where(np.array(filename_split) == 'mnt')[0][0]

    drive_letter = filename_split[mnt_i + 1].upper()

    remaining = '\\'.join(list(filename_split[mnt_i + 2:]))

    return '%s:\\%s' % (drive_letter, remaining)


def run_pipeline_js(pipeline, inputs, outputs, parameters=None, verbose=False, terminal_text_filename=None):

    command = 'ml-run-process %s ' % (pipeline)

    command += '--inputs '

    for key, value in inputs.items():
        if type(value) != list:
            command += '%s:%s ' % (str(key), str(value))
        else:
            for x in value:
                command += '%s:%s ' % (str(key), str(x))

    command += '--outputs '

    for key, value in outputs.items():
        command += '%s:%s ' % (str(key), str(value))

    if parameters is not None:
        command += '--parameters '

        for key, value in parameters.items():
            command += '%s:%s ' % (str(key), str(value))

    if terminal_text_filename is not None:
        # this will output the terminal text to the given filename
        command = command + '>> %s' % terminal_text_filename

    if verbose:
        print(command)

    profile = None
    cfg = BashConfigure()

    try:
        cfg.win32_wsl_open_bash('', [  # 'cd %s' % filepath, # needed for x-server to display visually
            # 'ls',
            command,
            'sleep 2'], profile)
    except PermissionError:
        cfg.win32_wsl_open_bash('', [  # 'cd %s' % filepath, # needed for x-server to display visually
            # 'ls',
            command,
            'sleep 2'], profile)


def run_sort(*, raw_fname=None, filt_fname=None, pre_fname=None, geom_fname=None, params_fname=None,
                 firings_out, filt_out_fname=None, pre_out_fname=None, metrics_out_fname=None, masked_out_fname=None,
                 freq_min=300, freq_max=7000, samplerate=30000, detect_sign=1,
                 adjacency_radius=-1, detect_threshold=3, detect_interval=10, clip_size=50,
                 firing_rate_thresh=0.05, isolation_thresh=0.95, noise_overlap_thresh=0.03,
                 peak_snr_thresh=1.5, mask_artifacts='true', whiten='true',
                 mask_threshold=6, mask_chunk_size=2000, terminal_text_filename=None,
                 mask_num_write_chunks=15, num_workers=os.cpu_count(), verbose=True):
    """
    Custom Sorting Pipeline. It will pre-process, sort, and curate (using ms_taggedcuration pipeline).

    Parameters
    ----------
    raw_fname : INPUT
        MxN raw timeseries array (M = #channels, N = #timepoints). If you input this it will pre-process the data.
    filt_fname : INPUT
        MxN raw timeseries array (M = #channels, N = #timepoints). This input contains data that has already been filtered.
    pre_fname : INPUT
        MxN pre-processed array timeseries array (M = #channels, N = #timepoints). This is if you want to analyze already pre-processed data.
    geom_fname : INPUT
        (Optional) geometry file (.csv format).
    params_fname : INPUT
        (Optional) parameter file (.json format), where the key is the any of the parameters for this pipeline. Any values in this .json file will overwrite any defaults.

    firings_out : OUTPUT
        The filename that will contain the spike data (.mda file), default to '/firings.mda'
    filt_out_fname : OUTPUT
        Optional filename for the filtered data (just filtered, no whitening).
    masked_out_fname : OUTPUT
        Optional filename for the masked_data.
    pre_out_fname : OUTPUT
        Optional filename for the pre-processed data (filtered and whitened).
    metrics_out_fname : OUTPUT
        The optional  output filename (.json) for the metrics that will be computed for each unit.

    samplerate : float
        (Optional) The sampling rate in Hz
    freq_min : float
        (Optional) The lower endpoint of the frequency band (Hz)
    freq_max : float
        (Optional) The upper endpoint of the frequency band (Hz)
    adjacency_radius : float
        (Optional) Radius of local sorting neighborhood, corresponding to the geometry file (same units). 0 means each channel is sorted independently. -1 means all channels are included in every neighborhood.
    detect_sign : int
        (Optional) Use 1, -1, or 0 to detect positive peaks, negative peaks, or both, respectively
    detect_threshold : float
        (Optional) Threshold for event detection, corresponding to the input file. So if the input file is normalized to have noise standard deviation 1 (e.g., whitened), then this is in units of std. deviations away from the mean.
    detect_interval : int
        (Optional) The minimum number of timepoints between adjacent spikes detected in the same channel neighborhood.
    clip_size : int
        (Optional) Size of extracted clips or snippets, used throughout
    firing_rate_thresh : float64
        (Optional) firing rate must be above this
    isolation_thresh : float64
        (Optional) isolation must be above this
    noise_overlap_thresh : float64
        (Optional) noise_overlap_thresh must be below this
    peak_snr_thresh : float64
        (Optional) peak snr must be above this
    mask_artifacts : str
        (Optional) if set to 'true', it will mask the large amplitude artifacts, if 'false' it will not.
    whiten : str
        (Optional) if set to 'true', it will whiten the signal (assuming the input is raw_fname, if 'false' it will not.
    mask_threshold : int
        (Optional) Number of standard deviations away from the mean RSS for the chunk to be considered as artifact.
    mask_chunk_size: int
        This chunk size will be the number of samples that will be set to zero if the RSS of this chunk is above threshold.
    mask_num_write_chunks: int
        How many mask_chunks will be simultaneously written to mask_out_fname (default of 150).
    num_workers : int
        (Optional) Number of simultaneous workers (or processes). The default is multiprocessing.cpu_count().
    terminal_text_filename: str
        (Optional) if you want to output the command prompt values to a text file, set this value to a filename you want it saved to.
    """

    # the name of the pipeline we will be using
    pipeline = 'ms4_geoff.sort'

    inputs = {}

    if raw_fname:
        inputs['raw_fname'] = raw_fname
    elif filt_fname:
        inputs['filt_fname'] = filt_fname
    elif pre_fname:
        inputs['pre_fname'] = pre_fname

    if geom_fname:
        inputs['geom_fname'] = geom_fname

    if params_fname:
        inputs['params_fname'] = params_fname

    outputs = {}

    if filt_out_fname is not None:
        outputs['filt_out_fname'] = filt_out_fname

    if firings_out is not None:
        outputs['firings_out'] = firings_out

    if pre_out_fname is not None:
        outputs['pre_out_fname'] = pre_out_fname

    if metrics_out_fname is not None:
        outputs['metrics_out_fname'] = metrics_out_fname

    if masked_out_fname is not None:
        outputs['masked_out_fname'] = masked_out_fname

    parameters = {'freq_min': freq_min, 'freq_max': freq_max, 'samplerate': samplerate,
                  'detect_sign': detect_sign, 'adjacency_radius': adjacency_radius,
                  'detect_threshold': detect_threshold, 'detect_interval': detect_interval,
                  'clip_size': clip_size, 'firing_rate_thresh': firing_rate_thresh, 'isolation_thresh': isolation_thresh,
                  'noise_overlap_thresh': noise_overlap_thresh, 'peak_snr_thresh': peak_snr_thresh,
                  'mask_artifacts': mask_artifacts, 'mask_chunk_size': mask_chunk_size, 'mask_threshold': mask_threshold,
                  'mask_num_write_chunks': mask_num_write_chunks, 'num_workers': num_workers, 'whiten': whiten}

    run_pipeline_js(pipeline, inputs, outputs, parameters, verbose=verbose, terminal_text_filename=terminal_text_filename)


def sort_finished(terminal_output_filename, max_time=300):
    # wait for the terminal output to exist

    # max_time = max time in secoinds to wait

    while not os.path.exists(terminal_output_filename):
        time.sleep(0.1)

    sort_invalid_string = ['Process returned with non-zero exit code']

    finished_string = '%s%s%s' % ('[ Saving to process cache ... ]\n',
                                  '[ Removing temporary directory ... ]\n',
                                  '[ Done. ]\n')

    sort_string = 'ms4_geoff_spec.py.mp ms4_geoff.sort'

    file_sorted = False
    finished = False
    start_time_set = False
    start_sort_time = None

    while not finished:
        with open(terminal_output_filename, 'r') as f:
            output_text = ''.join(f.readlines())  # want to make sure that the output is in text not a list

        if sort_string in output_text and not file_sorted and not start_time_set:
            start_sort_time = time.time()
            start_time_set = True

            file_length = len(output_text)

        while start_sort_time is None:
            time.sleep(0.1)
            continue

        if time.time() - start_sort_time >= max_time:
            # we've waited long enough for the file to sort.
            return False, "Retry"

        # check if the sort was broken.
        for invalid_str in sort_invalid_string:
            if invalid_str in output_text:
                return False, 'Abort'

        if len(output_text) == file_length:
            continue
        else:
            file_sorted = True

        if finished_string in output_text:
            finished = True

    return True, 'Complete'


def sort_intan(directory, tint_fullpath, Fs, whiten='true', detect_interval=10, detect_sign=0, detect_threshold=3,
             freq_min=300, freq_max=6000, mask_threshold=6, masked_chunk_size=None, mask_num_write_chunks=100,
             clip_size=50, self=None):

    tint_basename = os.path.basename(tint_fullpath)

    # set_filename = '%s.set' % tint_fullpath

    raw_fnames = [os.path.join(directory, file) for file in os.listdir(
        directory) if '_raw.mda' in file if tint_basename in file]

    for file in raw_fnames:
        mda_basename = os.path.splitext(file)[0]
        mda_basename = mda_basename[:find_sub(mda_basename, '_')[-1]]

        masked_out_fname = get_ubuntu_path(mda_basename + '_masked.mda')
        firings_out = get_ubuntu_path(mda_basename + '_firings.mda')
        filt_out_fname = get_ubuntu_path(mda_basename + '_filt.mda')
        pre_out_fname = get_ubuntu_path(mda_basename + '_pre.mda')
        metrics_out_fname = get_ubuntu_path(mda_basename + '_metrics.json')

        # check if these outputs have already been created, skip if they have
        existing_files = 0
        output_files = [masked_out_fname, filt_out_fname, firings_out, pre_out_fname, metrics_out_fname]
        for outfile in output_files:
            if os.path.exists(get_windows_filename(outfile)):
                existing_files += 1

        if existing_files == len(output_files):
            msg = '[%s %s]: The following file has already been sorted: %s, skipping sort!#Red' % \
                  (str(datetime.datetime.now().date()),
                   str(datetime.datetime.now().time())[:8], file)

            if self:
                self.LogAppend.myGUI_signal_str.emit(msg)
            else:
                print(msg)
            continue

        terminal_text_filename = get_ubuntu_path(mda_basename + '_terminal.txt')

        raw_fname = get_ubuntu_path(file)

        # Fs = int(get_setfile_parameter('rawRate', set_filename))

        if masked_chunk_size is None:
            masked_chunk_size = int(Fs/10)

        run_sort(raw_fname=raw_fname, filt_out_fname=filt_out_fname, pre_out_fname=pre_out_fname,
                 metrics_out_fname=metrics_out_fname, firings_out=firings_out, masked_out_fname=masked_out_fname,
                 samplerate=Fs, detect_interval=detect_interval, detect_sign=detect_sign,
                 detect_threshold=detect_threshold, freq_min=freq_min, freq_max=freq_max, mask_threshold=mask_threshold,
                 mask_chunk_size=masked_chunk_size, mask_num_write_chunks=mask_num_write_chunks, whiten=whiten,
                 clip_size=clip_size, terminal_text_filename=terminal_text_filename)

        # wait for the sort to finish before continuing
        finished, sort_code = sort_finished(get_windows_filename(terminal_text_filename))

        if 'Abort' not in sort_code:
            # re-try the sort if Abort is not in the code
            if not finished:
                os.remove(get_windows_filename(terminal_text_filename))

                run_sort(raw_fname=raw_fname, filt_out_fname=filt_out_fname, pre_out_fname=pre_out_fname,
                         metrics_out_fname=metrics_out_fname, firings_out=firings_out, masked_out_fname=masked_out_fname,
                         samplerate=Fs, detect_interval=detect_interval, detect_sign=detect_sign,
                         detect_threshold=detect_threshold, freq_min=freq_min, freq_max=freq_max,
                         mask_threshold=mask_threshold,
                         mask_chunk_size=masked_chunk_size, mask_num_write_chunks=mask_num_write_chunks, whiten=whiten,
                         clip_size=clip_size, terminal_text_filename=terminal_text_filename)

        else:
            msg = '[%s %s]: There was an error sorting the following file, consult terminal text file: %s!#Red' % \
                  (str(datetime.datetime.now().date()),
                   str(datetime.datetime.now().time())[:8], raw_fname)

            if self:
                self.LogAppend.myGUI_signal_str.emit(msg)
            else:
                print(msg)
            continue
