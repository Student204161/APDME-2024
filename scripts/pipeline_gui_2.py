import sys
import os
import json  # For converting dictionary to JSON string
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QSlider, QPushButton, QLineEdit, QFormLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap


class ScriptRunnerApp(QMainWindow):
    def __init__(self, obj_name, cam_thresh, scale_thresh, rmv_dist, reproj_masks, white_background):
        super().__init__()

        self.setWindowTitle("ImageViewer")
        self.setGeometry(1920, 1080, 1920, 1080)
        self.cam_thresh = cam_thresh
        self.scale_thresh = scale_thresh
        self.rmv_dist = float(rmv_dist)
        self.reproj_masks = reproj_masks
        self.white_background = white_background
        self.breakout = 0  # Initialize breakout variable
        self.segment = 0  # Initialize segment variable

        # Store folder path and initialize variables
        self.obj_name = obj_name
        if self.white_background:
            self.folder_path = os.path.join(
            f'data/GS_models/full_scene_MVG-{self.reproj_masks}_{self.cam_thresh}_RMVXYZ_{self.scale_thresh}_RMV{self.rmv_dist}STD_white',
            'test_' + obj_name, '1'
        )
        else:
            self.folder_path = os.path.join(
            f'data/GS_models/full_scene_MVG-{self.reproj_masks}_{self.cam_thresh}_RMVXYZ_{self.scale_thresh}_RMV{self.rmv_dist}STD',
            'test_' + obj_name, '1'
            )

        self.image_files = []
        self.current_image_index = 0

        self.init_ui()

    def init_ui(self):
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        explanation_label = QLabel(
            "Choose optimal parameters for filtering, using the novel render views as guidance (use slider or left-right arrow keys):\n"
            "1. The first parameter (Gamma) controls fraction of how much something has to be included in the masks to not be removed\n"
            "2. The second parameter (Size Thresh) removes big blobs that are higher than this value \n"
            "3. The third parameter (RMV Dist) removes blobs that are far away from where most blobs are"
        )
        explanation_label.setWordWrap(True)
        explanation_label.setAlignment(Qt.AlignTop)
        layout.addWidget(explanation_label)

        # Editable fields
        form_layout = QFormLayout()
        self.cam_thresh_field = QLineEdit(str(self.cam_thresh))
        self.scale_thresh_field = QLineEdit(str(self.scale_thresh))
        self.rmv_dist_field = QLineEdit(str(self.rmv_dist))
        form_layout.addRow("Cam Thresh:", self.cam_thresh_field)
        form_layout.addRow("Scale Thresh:", self.scale_thresh_field)
        form_layout.addRow("RMV Dist:", self.rmv_dist_field)
        layout.addLayout(form_layout)
        form_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Image display label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Slider for toggling images
        self.image_slider = QSlider(Qt.Horizontal, self)
        self.image_slider.setMinimum(0)
        self.image_slider.valueChanged.connect(self.update_image_from_slider)
        layout.addWidget(self.image_slider)

        # Exit button
        exit_button = QPushButton("Redo with other parameters", self)
        exit_button.clicked.connect(self.exit_and_return_data)
        layout.addWidget(exit_button)

        # Segment button
        segment_button = QPushButton("edit segmentation masks + Rerun reprojection", self)
        segment_button.clicked.connect(self.set_segment)
        layout.addWidget(segment_button)

        # Breakout button
        breakout_button = QPushButton("Exit", self)
        breakout_button.clicked.connect(self.set_breakout)
        layout.addWidget(breakout_button)


        # Load images from the folder
        self.display_images_from_folder(self.folder_path)

    def display_images_from_folder(self, folder_path):
        # Get list of image files from the folder passed as argument
        self.image_files = self.get_image_files_from_folder(folder_path)
        if self.image_files:
            # Configure slider and show the first image
            self.image_slider.setMaximum(len(self.image_files) - 1)
            self.show_image(self.image_files[self.current_image_index])
        else:
            self.image_label.setText("No images found in the folder.")
            self.image_slider.setDisabled(True)

    def get_image_files_from_folder(self, folder_path):
        # Get a list of all image files (jpg, png, etc.)
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]
        return sorted(image_files)  # Optional: sort alphabetically

    def show_image(self, image_path):
        # Display the image in the QLabel
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(1920//2, 
                                    1080//2, 
                                    Qt.KeepAspectRatio, 
                                    Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)


    def update_image_from_slider(self, value):
        # Update the current image index and display the corresponding image
        self.current_image_index = value
        self.show_image(self.image_files[self.current_image_index])

    def set_breakout(self):
        # Set breakout variable to True
        self.breakout = 1
        self.exit_and_return_data()

    def set_segment(self):
        # Set segment variable to True
        self.segment = 1
        self.exit_and_return_data()

    def exit_and_return_data(self):
        # Update values from the editable fields
        try:
            self.cam_thresh = float(self.cam_thresh_field.text())
            self.scale_thresh = self.scale_thresh_field.text()
            self.rmv_dist = self.rmv_dist_field.text()
        except ValueError:
            # Handle invalid inputs gracefully
            print("Invalid input detected. Using current values.")

        # Create the dictionary
        return_data = {
            "obj_name": self.obj_name,
            "cam_thresh": self.cam_thresh,
            "scale_thresh": self.scale_thresh,
            "rmv_dist": self.rmv_dist,
            "current_image_index": self.current_image_index,
            "breakout": self.breakout,
            "segment": self.segment
        }

        # Output the dictionary as a JSON string to stdout
        print(json.dumps(return_data))

        # Exit the application
        self.close()


if __name__ == "__main__":
    # Check if folder path is passed as an argument
    if len(sys.argv) < 5:
        print("Usage: python script.py <obj_name> <cam_thresh> <scale_thresh> <rmv_dist>")
        sys.exit(1)

    # Get the folder path from command-line arguments
    obj_name = sys.argv[1]
    cam_thresh = float(sys.argv[2])
    scale_thresh = sys.argv[3]
    rmv_dist = sys.argv[4]
    reproj_masks = sys.argv[5]
    white_background = sys.argv[6]

    app = QApplication(sys.argv)
    window = ScriptRunnerApp(obj_name, cam_thresh, scale_thresh, rmv_dist, reproj_masks, white_background)
    window.show()
    sys.exit(app.exec())
