import os
from PyQt5 import QtWidgets
import time
from core.utils import find_sub
from core.intan_rhd_functions import is_session_beginning, find_basename_files
from core.default_parameters import eeg_channels
from core.intan_mountainsort import validate_session


def addSessions(self):
    """Adds any sessions that are not already on the list"""

    while self.reordering_queue:
        # pauses add Sessions when the individual is reordering
        time.sleep(0.1)

    current_directory = self.current_directory_name

    if self.nonbatch == 0:
        try:
            sub_directories = [d for d in os.listdir(current_directory)
                               if os.path.isdir(os.path.join(current_directory, d)) and d not in ['Processed',
                                                                                                  'Converted']]
        except PermissionError:
            return
    else:
        sub_directories = [os.path.basename(current_directory)]
        current_directory = os.path.dirname(current_directory)

    added_rhd_files = []

    iterator = QtWidgets.QTreeWidgetItemIterator(self.directory_queue)
    # loops through all the already added sessions
    added_directories = []
    while iterator.value():
        directory_item = iterator.value()

        # check if the path still exists
        if not os.path.exists(os.path.join(current_directory, directory_item.data(0, 0))):
            # then remove from the list since it doesn't exist anymore
            root = self.directory_queue.invisibleRootItem()
            for child_index in range(root.childCount()):
                if root.child(child_index) == directory_item:
                    self.RemoveChildItem.myGUI_signal_QTreeWidgetItem.emit(directory_item)
        else:
            try:
                added_directories.append(directory_item.data(0, 0))
            except RuntimeError:
                # prevents issues where the object was deleted before it could be added
                return

        iterator += 1

    for directory in sub_directories:

        try:
            rhd_files = [os.path.join(current_directory, directory, file) for file in os.listdir(
                os.path.join(current_directory, directory)) if '.rhd' in file]
        except FileNotFoundError:
            return

        if rhd_files:
            if directory in added_directories:
                # add sessions that aren't already added
                # find the treewidget item
                iterator = QtWidgets.QTreeWidgetItemIterator(self.directory_queue)
                while iterator.value():
                    directory_item = iterator.value()
                    if directory_item.data(0, 0) == directory:
                        break
                    iterator += 1

                # find added rhd_files
                try:
                    iterator = QtWidgets.QTreeWidgetItemIterator(directory_item)
                except UnboundLocalError:
                    return
                except RuntimeError:
                    return

                while iterator.value():
                    session_item = iterator.value()
                    added_rhd_files.append(session_item.data(0, 0))
                    iterator += 1

                for rhd_file in rhd_files:
                    if rhd_file in added_rhd_files:
                        continue

                    added_rhd_files = addSession(self, rhd_file, current_directory, directory, added_rhd_files,
                                                 directory_item)

            else:
                # the directory has not been added yet
                directory_item = QtWidgets.QTreeWidgetItem()
                directory_item.setText(0, directory)

                for rhd_file in rhd_files:

                    if rhd_file in added_rhd_files:
                        continue

                    added_rhd_files = addSession(self, rhd_file, current_directory, directory, added_rhd_files,
                                                 directory_item)


def addSession(self, rhd_file, current_directory, directory, added_rhd_files, directory_item):

    directory = os.path.join(current_directory, directory)

    if is_session_beginning(rhd_file):
        tint_basename = os.path.basename(os.path.splitext(rhd_file)[0])
        tint_fullpath = os.path.join(directory, tint_basename)
        output_basename = '%s_ms' % tint_fullpath

        session_valid = validate_session(rhd_file, output_basename, eeg_channels, verbose=False)
        if session_valid and tint_basename != self.current_session:
            rhd_session_file = os.path.splitext(os.path.basename(rhd_file))[0]
            rhd_basename = rhd_session_file[:find_sub(rhd_session_file, '_')[-2]]
            session_files = find_basename_files(rhd_basename, directory)
            rhd_session_fullfile = os.path.join(directory, rhd_session_file + '.rhd')

            # find the session with our rhd file in it
            session_files = \
                [sub_list for sub_list in session_files if rhd_session_fullfile in sub_list][0]

            if type(session_files) != list:
                # if there is only one file in the list, the output will not be a list
                session_files = [session_files]

            # only addds the sessions that haven't been added already
            session_item = QtWidgets.QTreeWidgetItem()
            session_item.setText(0, tint_basename)

            for file in session_files:
                session_file_item = QtWidgets.QTreeWidgetItem()
                session_file_item.setText(0, file)
                session_item.addChild(session_file_item)

                added_rhd_files.append(file)

            directory_item.addChild(session_item)

            self.directory_queue.addTopLevelItem(directory_item)

    return added_rhd_files


def RepeatAddSessions(self):
    """This will continuously look for files to add to the Queue"""

    self.repeat_thread_active = True

    try:
        self.adding_session = True
        addSessions(self)
        # time.sleep(0.1)
        self.adding_session = False
    except FileNotFoundError:
        pass
    except RuntimeError:
        pass

    while True:

        if self.reset_add_thread:
            self.repeat_thread_active = False
            self.reset_add_thread = False
            return

        try:
            self.adding_session = True
            # time.sleep(0.1)
            addSessions(self)
            self.adding_session = False
            # time.sleep(0.1)
        except FileNotFoundError:
            pass
        except RuntimeError:
            pass
