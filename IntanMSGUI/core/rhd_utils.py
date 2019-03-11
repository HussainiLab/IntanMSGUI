import numpy as np

tetrode_map = {'buzsaki32': {1: [5, 4, 6, 3],
                             2: [13, 12, 14, 11],
                             3: [7, 2, 8, 1],
                             4: [15, 10, 16, 9],
                             5: [21, 20, 22, 19],
                             6: [29, 28, 30, 27],
                             7: [23, 18, 24, 17],
                             8: [31, 26, 32, 25],
                             },

               'buzsaki16': {1: [5, 4, 6, 3],
                             2: [13, 12, 14, 11],
                             3: [7, 2, 8, 1],
                             4: [15, 10, 16, 9],
                             },

               'axona16_angled': {1: [1, 2, 3, 4],
                                  2: [5, 6, 7, 8],
                                  3: [9, 10, 11, 12],
                                  4: [13, 14, 15, 16]
                                  },

               'axona16_new': {1: [15, 13, 11, 9],
                               2: [16, 14, 12, 10],
                               3: [1, 3, 5, 7],
                               4: [2, 4, 6, 8]
                               },

               'axona32': {1: [16, 14, 12, 10],
                           2: [15, 13, 11, 9],
                           3: [18, 20, 22, 24],
                           4: [17, 19, 21, 23],
                           5: [2, 4, 6, 8],
                           6: [1, 3, 5, 7],
                           7: [32, 30, 28, 26],
                           8: [31, 29, 27, 25],
                           },
               }


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


def intan_scalar():
    """returns the scalar value that can be element-wise multiplied to the data
    to convert from bits to micro-volts"""
    Vswing = 2.45
    bit_range = 2 ** 16  # it's 16 bit system
    gain = 192  # V/V, listed in the intan chip's datasheet
    return (1e6) * (Vswing) / (bit_range * gain)


def read_notes(notes, tetrode_map, default_probe='axona16_new'):
    probe = None
    experimenter = ''
    # reads through the notes to find useful information
    for note, note_val in notes.items():
        note_val = note_val.lower()
        if note_val != '':
            if note_val in list(tetrode_map.keys()):
                probe_list = list(tetrode_map.keys())
                probe = probe_list[probe_list.index(note_val)]
            elif 'probe:' in note_val:
                probe = note_val[note_val.find('probe:') + len('probe:'):].strip()
            else:
                experimenter = note_val

    if probe is None:
        probe = default_probe

    return probe, experimenter