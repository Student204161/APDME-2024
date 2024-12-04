import sys
import os
import subprocess
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QPushButton, QProgressBar, QComboBox
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal
import json

class FolderSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Folder Selector")
        self.setGeometry(200, 200, 1200, 900)

        self.init_ui()

    def init_ui(self):
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Label to display messages
        self.label = QLabel("Running task, please wait...")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Folder selection combo box (hidden initially)
        self.folder_selector = QComboBox(self)
        self.folder_selector.setVisible(False)
        layout.addWidget(self.folder_selector)

        # Button to confirm folder selection (hidden initially)
        self.select_button = QPushButton("Confirm Folder Selection", self)
        self.select_button.setVisible(False)
        self.select_button.clicked.connect(self.output_folder_path)
        layout.addWidget(self.select_button)

        # Start background task (run real script)
        self.run_real_script()

    def run_real_script(self):
        # Assuming 'from_mov_to_img.py' is the real Python script you want to run
        # You can modify this to point to the actual script path or location
        script_path = os.path.join(os.getcwd(), "scripts/data_preprocessing/from_mov_to_img.py")
        if not os.path.exists(script_path):
            self.label.setText(f"Script {script_path} not found.")
            return

        # Pass arguments to the script (e.g., working directory and 400 as an example argument)
        working_dir = os.getcwd()  # Example working directory, you can change it
        cmd = ['python', script_path, working_dir, '400']

        # Start the script as a subprocess
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        # Start reading the output and updating the progress bar
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_process_output)
        self.timer.start(100)  # Check every 100ms

    def check_process_output(self):
        # Check for output from the process
        output = self.process.stdout.readline()
        if output == '' and self.process.poll() is not None:
            # Process is done
            self.timer.stop()
            self.show_folders()  # Show folder selection after the task is complete
        elif output:
            # Extract progress information from output (assuming the script outputs progress in percentage)
            try:
                progress = int(output.strip())  # Example: the script should output progress as a number
                self.progress_bar.setValue(progress)
            except ValueError:
                pass  # Ignore if not a valid integer (i.e., if not progress-related)

    def show_folders(self):
        # Replace progress bar with folder selection UI
        self.label.setText("Select a folder from the list below:")
        self.progress_bar.setVisible(False)

        # Populate folder selection combo box
        self.folder_selector.setVisible(True)
        self.folder_selector.addItems(self.get_folders())
        self.select_button.setVisible(True)

    def get_folders(self):
        # Simulate fetching a list of folders from the working directory
        base_path = os.path.join(os.getcwd(), 'data/distorted_images/JPEGImages')  # Example: User's home directory
        return [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    def output_folder_path(self):
        selected_folder = self.folder_selector.currentText()
        if selected_folder:

            # Create the dictionary
            return_data = {
                "obj_name": selected_folder.split('/')[-1].split('.')[0]
            }

            # Output the dictionary as a JSON string to stdout
            print(json.dumps(return_data))

            self.label.setText(f"Folder chosen at ")
            self.close_application()

    def close_application(self):
        QApplication.quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FolderSelectorApp()
    window.show()
    sys.exit(app.exec())
