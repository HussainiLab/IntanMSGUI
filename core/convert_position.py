import os
import datetime
import struct
import numpy as np
import json
import core.intan_rhd_functions as load_rhd
from core.utils import session_datetime
import shutil


def MatlabNumSeq(start, stop, step, exclude=True):
    """In Matlab you can type:

    start:step:stop and easily create a numerical sequence

    if exclude is true it will exclude any values greater than the stop value
    """

    '''np.arange(start, stop, step) works good most of the time

    However, if the step (stop-start)/step is an integer, then the sequence
    will stop early'''

    seq = np.arange(start, stop + step, step)

    if exclude:
        if seq[-1] > stop:
            seq = seq[:-1]

    return seq


def pos2hz(t, x, y, start=None, stop=None, Fs=50):
    """This will convert the positions to 50 Hz values that"""
    if start is None:
        start = 0
    if stop is None:
        stop = np.amax(t)
    step = 1 / Fs  # 50 Hz sample rate
    post = MatlabNumSeq(start, stop, step, exclude=True)

    posx = np.zeros_like(post)
    posy = np.zeros_like(post)

    for i, t_value in enumerate(post):
        index = np.where(t <= t_value)[0][-1]
        posx[i] = x[index]
        posy[i] = y[index]

    return posx, posy, post


def rewrite_pos(session_files, positionSampleFreq, self=None):
    """
    This function will re-create the position files that were created by the VirtualMaze in the case that the .pos file
    was not created properly.
    """

    tint_basename = os.path.basename(os.path.splitext(session_files[-1])[0])
    directory = os.path.dirname(session_files[0])

    raw_filename = os.path.join(directory, tint_basename + '_raw_position.txt')
    cue_fname = os.path.join(directory, tint_basename + '_cues.json')
    pos_fpath = os.path.join(directory, tint_basename + '.pos')  # defining the name of the most recent .pos file

    with open(cue_fname, 'r') as f:
        settings = json.load(f)

    experimenter_name = settings['Experimenter Name:']
    arena = settings['Virtual Arena:']

    # read the position file

    if not os.path.exists(raw_filename):
        # then you cannot re-create the positions because the original position data was not saved
        # create a dummy position file
        return

    positions = np.loadtxt(raw_filename)
    positions[:, 0] = positions[:, 0] - positions[0, 0]

    t = positions[:, 0]
    x = positions[:, 1]
    y = positions[:, 2]

    # ------------------------------------

    file_header = load_rhd.read_header(session_files[0])  # read the file header information from a session file
    recording_Fs = int(file_header['sample_rate'])

    digital_input = False

    # check if there are digital inputs in the intan system that will be used to sync behavior w/ ephys data
    if file_header['num_board_dig_in_channels'] > 0:
        # then we have the digital values

        # reading the digital values #
        data_digital_in = np.array([])
        for session_file in sorted(session_files, reverse=False):
            # Loads each session and appends them to create one matrix of data for the current tetrode
            # file_data = load_rhd.read_data(session_file)  # loading the .rhd data
            file_data = load_rhd.read_data(session_file)

            # Acquiring session information

            if data_digital_in.shape[0] == 0:
                data_digital_in = file_data['board_dig_in_data']
            else:
                data_digital_in = np.concatenate((data_digital_in, file_data['board_dig_in_data']), axis=1)

        start_index, stop_index = load_rhd.get_data_limits(directory, tint_basename, data_digital_in, self=self)

        if start_index is not None and stop_index is not None:

            duration = np.floor((stop_index - start_index) / recording_Fs)
            start = 0
            stop = start + duration

            digital_input = True

            posx, posy, post = pos2hz(t, x, y, start=start, stop=stop - 1 / 50)

            positions = np.vstack((posx, posy, post)).T

            # the appropriate digital values were found, use the new get sync function
            window_values, original_positions, positions = sync_positions_with_pulse(positions, arena)

    if not digital_input:
        # then somehow the start and stop values were not recorded, use the old technique for synchronizing
        # the behavior with the ephys
        if 'maze_start_time' not in settings.keys():
            # we don't have the information we need, probably old data, create a dummy position file
            n_samples = int(len(positions))
            duration = int(n_samples / 50)
            dummy_parameters = get_dummy_parameters(arena, experimenter_name, duration)
            write_dummy_pos(pos_fpath, dummy_parameters)
            return

        maze_start_time = settings['maze_start_time']

        posx, posy, post = pos2hz(t, x, y)

        positions = np.vstack((posx, posy, post)).T

        window_values, original_positions, positions = sync_positions_with_ctime(positions, session_files,
                                                                                 maze_start_time,
                                                                                 positionSampleFreq, arena)

    pix_per_meter = settings['Pixels Per Meter(PPM): ']
    # min_x, max_x, min_y, max_y, window_min_x, window_max_x, window_min_y, window_max_y = window_values
    min_x, max_x, min_y, max_y, window_min_x, window_min_y, window_max_x, window_max_y = window_values
    try:
        n_samples = int(len(positions))
        duration = int(n_samples / 50)

        # --------------- write position file --------------
        with open(pos_fpath, 'wb+') as f:  # opening the .pos file

            trialdate, trialtime = load_rhd.session_datetime(pos_fpath)
            write_list = []
            header_vals = ['trial_data %s' % trialdate,
                      '\r\ntrial_time %s' % trialtime,
                      '\r\nexperimenter %s' % experimenter_name,
                      '\r\ncomments Arena:%s' % arena,
                      '\r\nduration %d' % duration,
                      '\r\nsw_version %s' % '1.3.0.16',
                      '\r\nnum_colours %d' % 4,
                      '\r\nmin_x %d' % min_x,
                      '\r\nmax_x %d' % max_x,
                      '\r\nmin_y %d' % min_y,
                      '\r\nmax_y %d' % max_y,
                      '\r\nwindow_min_x %d' % window_min_x,
                      '\r\nwindow_max_x %d' % window_max_x,
                      '\r\nwindow_min_y %d' % window_min_y,
                      '\r\nwindow_max_y %d' % window_max_y,
                      '\r\ntimebase %d hz' % 50,
                      '\r\nbytes_per_timestamp %d' % 4,
                      '\r\nsample_rate %.1f hz' % 50.0,
                      '\r\nEEG_samples_per_position %d' % 5,
                      '\r\nbearing_colour_1 %d' % 0,
                      '\r\nbearing_colour_2 %d' % 0,
                      '\r\nbearing_colour_3 %d' % 0,
                      '\r\nbearing_colour_4 %d' % 0,
                      '\r\npos_format t,x1,y1,x2,y2,numpix1,numpix2',
                      '\r\nbytes_per_coord %d' % 2,
                      '\r\npixels_per_metre %s' % pix_per_meter,
                      '\r\nnum_pos_samples %d' % n_samples,
                      '\r\ndata_start']

            for value in header_vals:
                write_list.append(bytes(value, 'utf-8'))

            onespot = 1  # this is just in case we decide to add other modes.

            # write_list = [bytes(headers, 'utf-8')]

            # write_list.append()
            for sample_num in np.arange(0, len(positions)):

                '''
                twospot => format: t,x1,y1,x2,y2,numpix1,numpix2
                onespot mode has the same format as two-spot mode except x2 and y2 take on values of 1023 (untracked value)

                note: timestamps and positions are big endian, i has 4 bytes, and h has 2 bytes
                '''

                if onespot == 1:
                    numpix1 = 1
                    numpix2 = 0
                    x2 = 1023
                    y2 = 1023
                    unused = 0
                    total_pix = numpix1  # total number of pixels tracked
                    write_list.append(struct.pack('>i', sample_num))

                    write_list.append(struct.pack('>8h', int(np.rint(positions[sample_num, 0])),
                                                  -int(np.rint(positions[sample_num, 1])), x2, y2, numpix1,
                                                  numpix2, total_pix, unused))

            write_list.append(bytes('\r\ndata_end\r\n', 'utf-8'))
            f.writelines(write_list)

            msg = '[%s %s]: Position values saved.' % (str(datetime.datetime.now().date()),
                                                                     str(datetime.datetime.now().time())[
                                                                     :8])
            if self is None:
                print(msg)
            else:
                self.mySrc2.myGUI_signal_str.emit(msg)

        with open(cue_fname, 'r') as f:
            settings = json.load(f)

        with open(cue_fname, 'w') as f:
            settings['duration'] = duration
            json.dump(settings, f)

    except PermissionError:
        '''re-writes the original pos_data if there's an error'''
        pass


def get_window_values(positions, arena):
    '''
    min_x = 0  # found in Tint's field view
    max_x = 512  # found in Tint's field view
    min_y = 0  # found in Tint's field view
    max_y = 523  # found in Tint's field view
    '''

    min_x = 0  # found in previous pos files
    max_x = 768  # found in previous pos files
    min_y = 0  # found in previous pos files
    max_y = 574  # found in previous pos files

    '''
    window_min_x = 284  # 768/2 - 100
    window_max_x = 484  # 768/2 + 100
    window_min_y = 187  # 574/2 - 100
    window_max_y = 387  # 574/2 + 100
    '''

    # I used to have hard coded values, but this is used in calculating the center
    # of the arena so I will code the center values here. Theoretically we could just
    # take the average of the min and max X and Y values and assume that is the center.
    # however it is possible that the mouse doesn't reach the outer edges of the map to
    # make this assumption valid.

    if any(arena == x for x in ['Simple Circular Track', 'Circular Track', 'Four Leaf Clover Track']):
        # outer radius of circle is 80 pixels
        window_min_x = 0
        window_min_y = 0
        window_max_x = 160
        window_max_y = 160

    elif arena == 'Linear Track':
        # linear track is 32 pixels wide and 400 pixels long
        window_min_x = 0
        window_min_y = 0
        window_max_x = 32
        window_max_y = 400

    else:
        print('The following arena has not been configured: %s, calculating center with behavior data.' % arena)
        window_min_x = 0
        window_min_y = 0

        xyrange = np.abs(np.amax(positions[:, :2], axis=0) - np.amin(positions[:, :2], axis=0))
        window_max_x = xyrange[0]
        window_max_y = xyrange[1]

    window_values = [min_x, max_x, min_y, max_y, window_min_x, window_min_y,
                     window_max_x, window_max_y]

    return window_values


def sync_positions_with_ctime(positions, current_session, maze_start_time, positionSampleFreq, arena):
    """This is the old way that we used to sync the positions with the ephys. When you record with the Intan software
    it will create the file right when you press 'Record' therefore we would just look at the time it was created and
    compare to the time point at which the behavior was started."""

    intan_start_time = os.path.getctime(current_session[-1])

    # -------------- center the arena ------------------------------------

    window_values = get_window_values(positions, arena)

    # center_vals = [np.mean([min_x, max_x]), np.mean([min_y, max_y])]
    # center_vals = np.amin(positions[:, :2], axis=0)

    original_positions = positions.copy()  # copying the original positions
    start_position = positions[0].reshape((1, len(positions[0])))
    # end_position = main.positions[-1].reshape((1, len(main.positions[-1])))

    # subtracting the means and then centering the values around the center points
    '''positions[:, :2] = positions[:, :2] - np.mean(positions[:, :2], axis=0) + np.array(
        [center_vals[0], center_vals[1]])'''

    # make sure the minimum x and y is zero
    positions[:, :2] = positions[:, :2] - np.amin(positions[:, :2], axis=0)

    # pl.plot(main.positions[:,0], main.positions[:,1], 'ro')

    # -------------- sync Axona with the GUI positions ---------------------
    if intan_start_time > maze_start_time:
        '''Intan was started after the GUI and thus we need to remove the first number of positions'''
        remove_samples = int((intan_start_time - maze_start_time) * positionSampleFreq)
        positions = positions[remove_samples - 1:]

    elif intan_start_time < maze_start_time:
        '''Intan was started before the GUI and thus the positions need to be padded with the start position'''
        missing_samples = int(-(intan_start_time - maze_start_time) * positionSampleFreq)
        positions = np.vstack((np.repeat(start_position, missing_samples, axis=0), positions))

    else:
        '''they have the same start_time, very unlikely'''
        pass

    return window_values, original_positions, positions


def sync_positions_with_pulse(positions, arena):
    # -------------- center the arena ------------------------------------

    window_values = get_window_values(positions, arena)

    # center_vals = [np.mean([min_x, max_x]), np.mean([min_y, max_y])]
    # center_vals = np.amin(positions[:, :2], axis=0)

    original_positions = positions.copy()  # copying the original positions

    # subtracting the means and then centering the values around the center points
    '''positions[:, :2] = positions[:, :2] - np.mean(positions[:, :2], axis=0) + np.array(
        [center_vals[0], center_vals[1]])'''

    # make sure the minimum x and y is zero
    positions[:, :2] = positions[:, :2] - np.amin(positions[:, :2], axis=0)

    return window_values, original_positions, positions


def get_dummy_parameters(arena, experimenter_name, duration):
    window_values = get_window_values()
    min_x, max_x, min_y, max_y, window_min_x, window_min_y, window_max_x, window_max_y = window_values
    dummy_session_parameters = {'experimenter': experimenter_name, 'arena': arena,
                                'min_x': min_x, 'max_x': max_x,
                                'min_y': min_y, 'max_y': max_y, 'window_min_x': window_min_x,
                                'window_max_x': window_max_x,
                                'window_min_y': window_min_y, 'window_max_y': window_max_y, 'ppm': 100,
                                'duration': duration}
    return dummy_session_parameters


def write_dummy_pos(position_file, session_parameters):
    position_samples = int(session_parameters['duration'] * 50)

    x = np.zeros((position_samples, 1))
    y = np.zeros_like(x)

    with open(position_file, 'wb+') as f:  # opening the .pos file
        trialdate, trialtime = session_datetime(position_file)
        write_list = []
        header_vals = ['trial_data %s' % trialdate,
                       '\r\ntrial_time %s' % trialtime,
                       '\r\nexperimenter %s' % session_parameters['experimenter'],
                       '\r\ncomments Arena:%s' % session_parameters['arena'],
                       '\r\nduration %d' % session_parameters['duration'],
                       '\r\nsw_version %s' % '1.3.0.16',
                       '\r\nnum_colours %d' % 4,
                       '\r\nmin_x %d' % session_parameters['min_x'],
                       '\r\nmax_x %d' % session_parameters['max_x'],
                       '\r\nmin_y %d' % session_parameters['min_y'],
                       '\r\nmax_y %d' % session_parameters['max_y'],
                       '\r\nwindow_min_x %d' % session_parameters['window_min_x'],
                       '\r\nwindow_max_x %d' % session_parameters['window_max_x'],
                       '\r\nwindow_min_y %d' % session_parameters['window_min_y'],
                       '\r\nwindow_max_y %d' % session_parameters['window_max_y'],
                       '\r\ntimebase %d hz' % 50,
                       '\r\nbytes_per_timestamp %d' % 4,
                       '\r\nsample_rate %.1f hz' % 50.0,
                       '\r\nEEG_samples_per_position %d' % 5,
                       '\r\nbearing_colour_1 %d' % 0,
                       '\r\nbearing_colour_2 %d' % 0,
                       '\r\nbearing_colour_3 %d' % 0,
                       '\r\nbearing_colour_4 %d' % 0,
                       '\r\npos_format t,x1,y1,x2,y2,numpix1,numpix2',
                       '\r\nbytes_per_coord %d' % 2,
                       '\r\npixels_per_metre %f' % 100,
                       '\r\nnum_pos_samples %d' % position_samples,
                       '\r\ndata_start']

        for value in header_vals:
            write_list.append(bytes(value, 'utf-8'))

        onespot = 1  # this is just in case we decide to add other modes.

        # write_list = [bytes(headers, 'utf-8')]

        # write_list.append()
        for sample_num in np.arange(0, len(x)):

            '''
            twospot => format: t,x1,y1,x2,y2,numpix1,numpix2
            onespot mode has the same format as two-spot mode except x2 and y2 take on values of 1023 (untracked value)

            note: timestamps and positions are big endian, i has 4 bytes, and h has 2 bytes
            '''

            if onespot == 1:
                numpix1 = 1
                numpix2 = 0
                x2 = 1023
                y2 = 1023
                unused = 0
                total_pix = numpix1  # total number of pixels tracked
                # t_byte = struct.pack('>i', sample_num)
                write_list.append(struct.pack('>i', sample_num))
                # pos_byte = struct.pack('>8h', int(np.rint(main.positions[sample_num,0])), int(
                # np.rint(main.positions[sample_num,1])), x2, y2, numpix1, numpix2, total_pix, unused)

                write_list.append(struct.pack('>8h', int(np.rint(x[sample_num])),
                                              int(np.rint(y[sample_num])), x2, y2, numpix1,
                                              numpix2, total_pix, unused))

        # f.seek(0, 2)
        # f.writelines([bytes(headers, 'utf-8'), t_byte, pos_byte, bytes('\r\ndata_end\r\n', 'utf-8')])
        write_list.append(bytes('\r\ndata_end\r\n', 'utf-8'))
        f.writelines(write_list)


def convert_position(session_files, position_filename, positionSampleFreq, output_basename, self=None):

    if not os.path.exists(position_filename):

        msg = '[%s %s]: Creating the following .pos file: %s!' % \
            (str(datetime.datetime.now().date()),
             str(datetime.datetime.now().time())[:8], position_filename)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)

            rewrite_pos(session_files, positionSampleFreq, self=self)

    else:

        msg = '[%s %s]: The following position file already exists: %s!' % \
              (str(datetime.datetime.now().date()),
               str(datetime.datetime.now().time())[:8], position_filename)

        if self is None:
            print(msg)
        else:
            self.LogAppend.myGUI_signal_str.emit(msg)

    output_pos_filename = '%s.pos' % output_basename
    if not os.path.exists(output_pos_filename):
        shutil.copy(position_filename, output_pos_filename)
