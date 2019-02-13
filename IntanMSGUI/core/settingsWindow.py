from PyQt5 import QtWidgets, QtCore

from core.gui_utils import background, gui_name, center
from core.default_parameters import pre_spike_samples, post_spike_samples, detect_sign, detect_threshold, \
    detect_interval, interpolation, desired_Fs, freq_min, freq_max, mask_threshold, mask, num_features, \
    max_num_clips_for_pca, flip_sign, software_rereference, reref_method, remove_outliers, remove_spike_percentage, \
    notch_filter, clip_scalar, remove_method


class SettingsWindow(QtWidgets.QWidget):  # defines the settings_window

    def __init__(self, main_window):  # initializes the settings window
        super(SettingsWindow, self).__init__()

        background(self)  # acquires some features from the background function we defined earlier

        self.main_window = main_window

        self.setWindowTitle("%s - Settings Window" % gui_name)  # sets the title of the window

        self.settings()

    def settings(self):
        # -------- settings widgets ------------------------

        # interpolation settings
        self.interpolate_cb = QtWidgets.QCheckBox("Interpolate")
        if interpolation:
            self.interpolate_cb.toggle()  # default it to curated

        self.interpolate_Fs = QtWidgets.QLineEdit()
        self.interpolate_Fs.setText(str(desired_Fs))
        self.interpolate_Fs.setAlignment(QtCore.Qt.AlignHCenter)

        interpolate_layout = QtWidgets.QHBoxLayout()
        interpolate_layout.addWidget(QtWidgets.QLabel("Interpolation Fs:"))
        interpolate_layout.addWidget(self.interpolate_Fs)

        # bandpass frequency values
        self.upper_cutoff = QtWidgets.QLineEdit()
        self.upper_cutoff.setText(str(freq_max))
        self.upper_cutoff.setAlignment(QtCore.Qt.AlignHCenter)
        upper_cutoff_layout = QtWidgets.QHBoxLayout()
        upper_cutoff_layout.addWidget(QtWidgets.QLabel("Upper Bandwidth (Hz):"))
        upper_cutoff_layout.addWidget(self.upper_cutoff)

        self.lower_cutoff = QtWidgets.QLineEdit()
        self.lower_cutoff.setText(str(freq_min))
        self.lower_cutoff.setAlignment(QtCore.Qt.AlignHCenter)
        lower_cutoff_layout = QtWidgets.QHBoxLayout()
        lower_cutoff_layout.addWidget(QtWidgets.QLabel("Lower Bandwidth (Hz):"))
        lower_cutoff_layout.addWidget(self.lower_cutoff)

        # spike detection parameters
        self.pre_threshold_widget = QtWidgets.QSpinBox()
        pre_threshold_layout = QtWidgets.QHBoxLayout()
        pre_threshold_layout.addWidget(QtWidgets.QLabel("Pre-Spike Samples:"))
        pre_threshold_layout.addWidget(self.pre_threshold_widget)
        self.pre_threshold_widget.setMinimum(0)
        self.pre_threshold_widget.setMaximum(50)
        self.pre_threshold_widget.setValue(pre_spike_samples)

        self.post_threshold_widget = QtWidgets.QSpinBox()
        post_threshold_layout = QtWidgets.QHBoxLayout()
        post_threshold_layout.addWidget(QtWidgets.QLabel("Post-Spike Samples:"))
        post_threshold_layout.addWidget(self.post_threshold_widget)
        self.post_threshold_widget.setMinimum(0)
        self.post_threshold_widget.setMaximum(50)
        self.post_threshold_widget.setValue(post_spike_samples)

        self.detect_sign_combo = QtWidgets.QComboBox()
        self.detect_sign_combo.addItem("Positive Peaks")
        self.detect_sign_combo.addItem("Negative Peaks")
        self.detect_sign_combo.addItem("Positive and Negative Peaks")
        self.detect_sign_combo.currentIndexChanged.connect(self.detect_sign_changed)
        self.detect_sign = detect_sign  # initializing detect_sign value

        text_value = None
        if detect_sign == 0:
            text_value = 'Positive and Negative Peaks'
        elif detect_sign == 1:
            text_value = 'Positive Peaks'
        elif detect_sign == -1:
            text_value = 'Negative Peaks'

        self.detect_sign_combo.setCurrentIndex(self.detect_sign_combo.findText(text_value))

        detect_sign_layout = QtWidgets.QHBoxLayout()
        detect_sign_layout.addWidget(QtWidgets.QLabel("Detect Sign:"))
        detect_sign_layout.addWidget(self.detect_sign_combo)

        self.detect_threshold_widget = QtWidgets.QLineEdit()
        detect_threshold_layout = QtWidgets.QHBoxLayout()
        detect_threshold_layout.addWidget(QtWidgets.QLabel("Detect Threshold:"))
        detect_threshold_layout.addWidget(self.detect_threshold_widget)
        self.detect_threshold_widget.setText(str(detect_threshold))
        self.detect_threshold_widget.setAlignment(QtCore.Qt.AlignHCenter)

        self.whiten_cb = QtWidgets.QCheckBox('Whiten')
        self.whiten_cb.toggle()  # set the default to whiten
        self.whiten_cb.stateChanged.connect(self.changed_whiten)

        self.detect_interval_widget = QtWidgets.QSpinBox()
        detect_interval_layout = QtWidgets.QHBoxLayout()
        detect_interval_layout.addWidget(QtWidgets.QLabel("Detect Interval:"))
        detect_interval_layout.addWidget(self.detect_interval_widget)
        self.detect_interval_widget.setValue(detect_interval)

        self.flip_sign = QtWidgets.QCheckBox('Flip Sign')
        if flip_sign:
            self.flip_sign.toggle()

        # masking widgets
        self.mask = QtWidgets.QCheckBox('Mask Artifacts')
        if mask:
            self.mask.toggle()

        self.mask_threshold = QtWidgets.QLineEdit()
        mask_threshold_layout = QtWidgets.QHBoxLayout()
        mask_threshold_layout.addWidget(QtWidgets.QLabel("Mask Threshold:"))
        mask_threshold_layout.addWidget(self.mask_threshold)
        self.mask_threshold.setText(str(mask_threshold))
        self.mask_threshold.setAlignment(QtCore.Qt.AlignHCenter)

        # PCA parameters
        self.num_features = QtWidgets.QLineEdit()
        num_features_layout = QtWidgets.QHBoxLayout()
        num_features_layout.addWidget(QtWidgets.QLabel("# of Features:"))
        num_features_layout.addWidget(self.num_features)
        self.num_features.setText(str(num_features))
        self.num_features.setAlignment(QtCore.Qt.AlignHCenter)

        self.max_num_clips_for_pca = QtWidgets.QLineEdit()
        max_num_clips_for_pca_layout = QtWidgets.QHBoxLayout()
        max_num_clips_for_pca_layout.addWidget(QtWidgets.QLabel("Max # Clips for PCA:"))
        max_num_clips_for_pca_layout.addWidget(self.max_num_clips_for_pca)
        self.max_num_clips_for_pca.setText(str(max_num_clips_for_pca))
        self.max_num_clips_for_pca.setAlignment(QtCore.Qt.AlignHCenter)

        # notch filter

        self.notch_filter = QtWidgets.QCheckBox("Notch Filter:")
        if notch_filter:
            self.notch_filter.toggle()

        # reref parameters
        self.software_rereference = QtWidgets.QCheckBox("Software Re-Reference")

        if software_rereference:
            self.software_rereference.toggle()

        self.reref_method_combo = QtWidgets.QComboBox()
        reref_method_layout = QtWidgets.QHBoxLayout()
        reref_method_layout.addWidget(QtWidgets.QLabel("Re-Ref Channel Method:"))
        reref_method_layout.addWidget(self.reref_method_combo)
        self.reref_method_combo.addItem("sd")
        self.reref_method_combo.addItem("avg")
        self.reref_method_combo.setCurrentIndex(self.reref_method_combo.findText(reref_method))

        # remove outlier parameters
        self.remove_outliers = QtWidgets.QCheckBox("Remove Outliers")
        if remove_outliers:
            self.remove_outliers.toggle()

        self.remove_outliers_percentage = QtWidgets.QLineEdit()
        remove_outliers_percentage_layout = QtWidgets.QHBoxLayout()
        remove_outliers_percentage_layout.addWidget(QtWidgets.QLabel("Remove Outlier Percentage(%):"))
        remove_outliers_percentage_layout.addWidget(self.remove_outliers_percentage)
        self.remove_outliers_percentage.setText(str(remove_spike_percentage))
        self.remove_outliers_percentage.setAlignment(QtCore.Qt.AlignHCenter)

        self.remove_method = QtWidgets.QComboBox()
        remove_method_layout = QtWidgets.QHBoxLayout()
        remove_method_layout.addWidget(QtWidgets.QLabel("Clipping Method:"))
        remove_method_layout.addWidget(self.remove_method)
        self.remove_method.addItem("max")
        self.remove_method.addItem("median")
        self.remove_method.addItem("mean")
        self.remove_method.setCurrentIndex(self.remove_method.findText(remove_method))

        self.okay_button = QtWidgets.QPushButton("Confirm")

        # adding the widgets to the main window
        ms_settings_layout = QtWidgets.QVBoxLayout()
        settings_layer1 = QtWidgets.QHBoxLayout()
        settings_layer2 = QtWidgets.QHBoxLayout()
        settings_layer3 = QtWidgets.QHBoxLayout()
        settings_layer4 = QtWidgets.QHBoxLayout()
        settings_layer5 = QtWidgets.QHBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()

        settings_layers = [settings_layer1, settings_layer2, settings_layer3, settings_layer4, settings_layer5,
                           button_layout]

        # for layer in settings_layers:
        #    layer.addStretch(1)

        layer1_widgets = [lower_cutoff_layout, upper_cutoff_layout, self.notch_filter]

        layer2_widgets = [pre_threshold_layout, post_threshold_layout, detect_sign_layout, detect_threshold_layout,
                          detect_interval_layout]

        layer3_widgets = [self.mask, mask_threshold_layout, num_features_layout,
                          max_num_clips_for_pca_layout]

        layer4_widgets = [self.software_rereference, reref_method_layout, self.remove_outliers,
                          remove_outliers_percentage_layout, remove_method_layout]

        layer5_widgets = [self.interpolate_cb, interpolate_layout, self.flip_sign, self.whiten_cb]

        button_widgets = [self.okay_button]

        layer_widgets = [layer1_widgets, layer2_widgets, layer3_widgets, layer4_widgets, layer5_widgets, button_widgets]

        for layer, widgets in zip(settings_layers, layer_widgets):
            for widget in widgets:
                if 'Layout' in str(widget):
                    layer.addLayout(widget)
                    # layer.addStretch(1)
                else:
                    layer.addWidget(widget)
                    # layer.addStretch(1)
        layer.addStretch(1)

        for layer in settings_layers:
            ms_settings_layout.addLayout(layer)

        self.setLayout(ms_settings_layout)  # sets the widget to the one we defined

        center(self)  # centers the widget on the screen

    def detect_sign_changed(self):

        current_value = self.detect_sign_combo.currentText()

        if 'Pos' in current_value and "Neg" in current_value:
            self.detect_sign = 0
        elif 'Pos' in current_value:
            self.detect_sign = 1
        elif 'Neg' in current_value:
            self.detect_sign = -1
        else:
            self.main_window.LogAppend.myGUI_signal_str.emit('detect sign value does not exist')

    def changed_whiten(self):
        self.main_window.analyzed_sessions = []
        self.main_window.directory_queue.clear()
        self.main_window.restart_add_sessions_thread()
