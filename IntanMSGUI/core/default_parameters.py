# ------------- default parameters not shown in GUI -----------------
"""
I felt like these parameters were just not worth exposing to the users via the GUI, feel free to change them
if you want.
"""
# EEG Settings

eeg_channels = 'first'  # this will save the 1st channel as an .eeg in each tetrode
# eeg_channels = 'all'  # this will save all the channels as their own .eeg file
# eeg_channels = [W, X, Y, Z]  # list of channels if you want to specify which ones to use

masked_chunk_size = None
# use a value of Fs/10 or Fs/20, I forget

mask_num_write_chunks = 100  # this is how many of the chunk segments will be written at the same time, mainly just for
# optimizing write speeds, just try to leave it as ~100k - 300k samples.

clip_size = 50  # samples per spike, default 50


# Axona Artifact Rejection Criteria, I'd just leave these. They are in the manual
rejthreshtail = 43  # I think, the latter 20-30% can't be above this value ( I think)
rejstart = 30  #
rejthreshupper = 100  # if 1st sample is above this value in bits, it will discard
rejthreshlower = -100  # if 1st sample is below this value in bits, it will discard

positionSampleFreq = 50  # sampling frequency of position, default 50

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

# ------------- default parameters that are shown within the GUI ---------------

# INTERPOLATE
interpolation = True  # if you want to interpolate, options: True or False, remember capital letter
desired_Fs = int(48e3)  # Sampling Frequency you will interpolate to

# WHITEN
whiten = 'true'  # if you want to whiten or not, 'true' or 'false', lower case letters
# whiten = 'false'

# THRESHOLD
flip_sign = True  # if you want to flip the signal (multiply by -1)

detect_interval = 30  # it will split up the data into segments of this value and find a peak/trough for each segment
detect_sign = 1  # 0 = positive and negative peaks, 1 = positive peaks, -1 = negative peaks

if whiten == 'true':
    # the threshold of the data, if whitened, data is normalized to standard deviations, so a value of 3
    detect_threshold = 3.5
    # would mean 3 standard deviations away from baseline. If not whitened treat it like a Tint threshold, i.e. X bits.
    # essentially bits vs standard deviations.
else:
    # you can grab a value from a set file with this animal, the parameter is called 'threshold', it is however in
    # 16bit, so divide the value by 256 to convert to 8bit, since the thresholding is in 8 bit.
    detect_threshold = 33

# BANDPASS
freq_min = 300  # min frequency to bandpass at
freq_max = 6000  # max freq to bandpass at

# MASK
# The amount of indices that you will segment for masking artifacts. if you leave as None it will
# mask = True
mask = True

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

# if you want to add a notch. However, it is already filtered using freq_min, so this doesn't really help
notch_filter = True
# unless of course your freqmin is below 60 Hz, default False

pre_spike_samples = 15  # number of samples pre-threshold samples to take, default 10
post_spike_samples = 35  # number of post threshold samples to take, default 40

if pre_spike_samples + post_spike_samples != clip_size:
    raise ValueError("the pre (%d) and post (%d) spike samples need to add up to the clip_size: %d." % (pre_spike_samples,
                                                                                                        post_spike_samples,
                                                                                                        int(clip_size)))

# The percentage of spikes to remove as outliers, this will make it so they don't make the majority of the
# spikes look really small in amplitude
remove_outliers = True  # if False, it won't remove outliers, if True, it will.
remove_spike_percentage = 5  # percent value, default 1, haven't experimented much with this

remove_method = 'max'  # this will find the max of the peak values (or min if it's negative)
# and set that as the clipping value

clip_scalar = 1.05  # this will multiply the clipping value found via the remove_method method,
# and then scale by this value.

# feature parameters
num_features = 10
max_num_clips_for_pca = 3000


# END PARAMETERS
