from krita import DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Krita, ManagedColor
from PyQt5.QtCore import Qt, QPoint, QSize, QEvent, QRect, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTabWidget, QSlider, QLabel,
    QFrame, QPushButton, QButtonGroup, QRadioButton, QFileDialog,
    QHBoxLayout, QGridLayout, QSplitter, QScrollArea, QDialog, QGroupBox, QSizePolicy
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPainterPath, QConicalGradient, QBrush
import os
import sys
sys.path.append(os.path.expanduser("~/ddraw/lib/python3.10/site-packages"))
import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
import os
import time
import sys
from .category_data import ValueData, ColorData

class ValueButton(QPushButton):
    def __init__(self, value, hex_code, is_reference=False, parent=None):
        super().__init__(parent)
        self.value = value
        self.hex_code = hex_code
        self.is_reference = is_reference
        self.matched_button = None
        self.setMinimumSize(60, 60)
        self.setMaximumSize(60, 60)
        border_style = "2px solid #00FF00" if is_reference else "1px solid #888888"
        self.setStyleSheet(f"background-color: {hex_code}; border: {border_style};")
        self.setToolTip(hex_code)
        
    def set_matched_button(self, button):
        self.matched_button = button

class ValuePairWidget(QWidget):
    """Display a matched canvas–reference value pair with interactive buttons."""
    def __init__(self, canvas_rgb, canvas_hex, ref_rgb, ref_hex, parent=None):
        super().__init__(parent)
        self.canvas_hex = canvas_hex
        self.ref_hex = ref_hex
        
        layout = QHBoxLayout()
        layout.setSpacing(5)
        self.setLayout(layout)
        
        # Create buttons
        self.canvas_button = ValueButton(canvas_rgb, canvas_hex, is_reference=False)
        self.ref_button = ValueButton(ref_rgb, ref_hex, is_reference=True)
        self.canvas_button.set_matched_button(self.ref_button)
        self.ref_button.set_matched_button(self.canvas_button)
        
        # Labels
        canvas_label = QLabel(f"{canvas_rgb}")
        canvas_label.setAlignment(Qt.AlignCenter)
        canvas_label.setStyleSheet("color: white; background-color: #333333; padding: 2px;")
        
        ref_label = QLabel(f"{ref_rgb}")
        ref_label.setAlignment(Qt.AlignCenter)
        ref_label.setStyleSheet("color: white; background-color: #333333; padding: 2px;")
        arrow_label = QLabel("→")
        arrow_label.setStyleSheet("color: #FFFF00; font-size: 16px; font-weight: bold;")
        arrow_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        layout.addWidget(self.canvas_button)
        layout.addWidget(canvas_label)
        layout.addWidget(arrow_label)
        layout.addWidget(ref_label)
        layout.addWidget(self.ref_button)

class ValueColor(QWidget):
    """Widget for loading images, applying filters, and showing value/color analyses."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value_image = None
        self.color_image = None
        self.value_reference_image = None
        self.color_reference_image = None
        self.canvas_image = None
        self.current_filter = None
       
        self.value_data = ValueData()
        self.color_data = ColorData()
        
        self.value_pair_widgets = []
        self.color_pair_widgets = []
        self.selectionTimer = QTimer()
        self.selectionTimer.setSingleShot(True)
        self.selectionTimer.timeout.connect(self.checkSelection)

    def _make_preview_section(self, prefix, left_text, right_text):
        """
        Creates:
          - self.{prefix}_preview_container
          - QHBoxLayout on it
          - self.{prefix}_left_preview_label
          - self.{prefix}_right_preview_label
          - self.{prefix}_preview_splitter
        Returns the container widget (so caller can add it to layout).
        """
        container = QWidget()
        layout = QHBoxLayout(container)

        left = QLabel(left_text)
        left.setAlignment(Qt.AlignCenter)
        left.setStyleSheet("background-color: black; color: white;")
        left.setMinimumHeight(300)
        setattr(self, f"{prefix}_left_preview_label", left)

        right = QLabel(right_text)
        right.setAlignment(Qt.AlignCenter)
        right.setStyleSheet("background-color: black; color: white;")
        right.setMinimumHeight(300)
        setattr(self, f"{prefix}_right_preview_label", right)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([200,200])
        setattr(self, f"{prefix}_preview_splitter", splitter)

        layout.addWidget(splitter)
        setattr(self, f"{prefix}_preview_container", container)

        return container

    def _make_pairs_section(self, prefix, header_text):
        """
        Creates:
          - self.{prefix}_pairs_layout (QVBoxLayout)
          - self.{prefix}_matched_pairs_label (QLabel header)
          - self.{prefix}_pairs_container (+ its layout)
          - QScrollArea containing that container.
        Returns the QLayout (so caller can add it to the tab).
        """
        pairs_layout = QVBoxLayout()
        setattr(self, f"{prefix}_pairs_layout", pairs_layout)

        header = QLabel(header_text)
        header.setStyleSheet("font-weight: bold; font-size: 14px;")
        setattr(self, f"{prefix}_matched_pairs_label", header)
        pairs_layout.addWidget(header)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        setattr(self, f"{prefix}_pairs_container", container)
        setattr(self, f"{prefix}_pairs_container_layout", container_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setMinimumHeight(200)
        scroll.setMaximumHeight(400)
        pairs_layout.addWidget(scroll)

        return pairs_layout

    def _make_feedback_label(self, prefix, text):
        """
        Creates self.{prefix}_feedback_label with consistent stylesheet
        and returns it.
        """
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignLeft)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("""
            QLabel {
                color: white;
                background-color: #333333;
                padding: 2px;
                border-radius: 2px;
                margin: 2px;
            }
        """)
        setattr(self, f"{prefix}_feedback_label", lbl)
        return lbl

    # Individual tab-creators
    def create_value_tab(self):
        """Create the tab for value analysis"""
        self.value_tab = QWidget()
        layout = QVBoxLayout(self.value_tab)

        # canvas picker
        btn = QPushButton("Set Current Canvas")
        btn.clicked.connect(self.show_current_canvas)
        layout.addWidget(btn)

        # filter radios + slider
        filter_group = QGroupBox("Filter Options")
        h = QHBoxLayout(filter_group)
        self.filter_group = QButtonGroup(filter_group)
        for name in ("Gaussian","Bilateral","Median"):
            rb = QRadioButton(name)
            # re-create the old attributes so upload_image() still works:
            if name == "Gaussian":
                self.gaussian_radio = rb
            elif name == "Bilateral":
                self.bilateral_radio = rb
            else:
                self.median_radio = rb

            rb.clicked.connect(lambda _, n=name.lower(): self.filter_selected(n))
            self.filter_group.addButton(rb)
            h.addWidget(rb)
        layout.addWidget(filter_group)

        self.slider_label = QLabel("Kernel Size (%): 1.5%")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(15,49)
        self.slider.setValue(15)
        self.slider.valueChanged.connect(self.update_kernel_size_label)
        self.slider.valueChanged.connect(self.update_preview)
        self.slider_label.hide()
        self.slider.hide()
        layout.addWidget(self.slider_label)
        layout.addWidget(self.slider)

        # --- SHARED SECTIONS ---
        layout.addWidget(self._make_preview_section(
            "value", "Canvas", "Reference"
        ))
        fb_btn = QPushButton("Get Value Feedback")
        fb_btn.clicked.connect(self.get_feedback_value)
        self.value_feedback_btn = fb_btn
        layout.addWidget(fb_btn)

        layout.addLayout(self._make_pairs_section(
            "value", "Value Pairs (Canvas → Reference):"
        ))

        layout.addWidget(self._make_feedback_label(
            "value", "Click 'Get Value Feedback' to analyze the canvas values"
        ))

    def create_color_tab(self):
        """Create the tab for color analysis"""
        self.color_tab = QWidget()
        layout = QVBoxLayout(self.color_tab)
        layout.setAlignment(Qt.AlignTop)
        self.setWindowTitle("Color Cluster Matcher")

        # Process & zoom controls
        proc = QPushButton("Process Reference Image")
        proc.clicked.connect(self.process_reference_image)
        layout.addWidget(proc)

        zoom_h = QHBoxLayout()
        for txt, slot in (("- Zoom out", self.zoom_out),("+ Zoom in", self.zoom_in)):
            btn = QPushButton(txt)
            btn.clicked.connect(slot)
            zoom_h.addWidget(btn)
        layout.addLayout(zoom_h)

        # scrollable image container
        self.scroll_area = QScrollArea(self.color_tab)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        img_cont = QWidget()
        v = QVBoxLayout(img_cont)
        self.image_label = ClusterHoverLabel(img_cont)
        self.image_label.setMinimumSize(200,200)
        self.image_label.clusterHovered.connect(self.update_cluster_info)
        v.addWidget(self.image_label)

        self.scroll_area.setWidget(img_cont)
        layout.addWidget(self.scroll_area)

        # Color tools & fill options
        tools = QGroupBox("Color Tools")
        tv = QVBoxLayout(tools)
        for txt, slot in (("Select Color",self.selectColor),
                          ("Lasso Fill Tool",self.activateLassoTool)):
            btn = QPushButton(txt)
            btn.clicked.connect(slot)
            tv.addWidget(btn)

        fill_grp = QGroupBox("Fill Options")
        fv = QVBoxLayout(fill_grp)
        fc = QPushButton("Select Fill Color")
        fc.clicked.connect(self.selectFillColor)
        fc.setStyleSheet("background-color: #ffffff;")
        fb = QPushButton("Fill Selection")
        fb.clicked.connect(self.fillSelection)
        fb.setEnabled(False)
        fv.addWidget(fc)
        fv.addWidget(fb)
        fill_grp.setLayout(fv)
        fill_grp.setVisible(False)

        tv.addWidget(fill_grp)
        layout.addWidget(tools)

        # --- SHARED SECTIONS ---
        layout.addWidget(self._make_preview_section(
            "color", "Canvas", "Reference"
        ))
        cfbtn = QPushButton("Get Color Feedback")
        cfbtn.clicked.connect(self.get_feedback_color)
        self.color_feedback_btn = cfbtn
        layout.addWidget(cfbtn)

        layout.addLayout(self._make_pairs_section(
            "color", "Color Pairs (Canvas → Reference):"
        ))

        layout.addWidget(self._make_feedback_label(
            "color", "Process reference image first"
        ))

        # Initialize image data storage
        self.current_image = None
        self.current_labels = None
        self.current_colors = None
        self.current_groups = None
        
    ## Color hover 
    def process_reference_image(self):
        """Process the stored reference image for color analysis"""
        if hasattr(self, 'color_reference_image') and self.color_reference_image is not None:
            # Convert color space appropriately
            if len(self.color_reference_image.shape) == 2:  # Grayscale
                self.current_image = cv2.cvtColor(self.color_reference_image, cv2.COLOR_GRAY2RGB)
            elif self.color_reference_image.shape[2] == 4:  # RGBA
                self.current_image = cv2.cvtColor(self.color_reference_image, cv2.COLOR_BGRA2RGB)
            else:  # BGR
                self.current_image = cv2.cvtColor(self.color_reference_image, cv2.COLOR_BGR2RGB)
            
            self.update_cluster_count()
             
    def update_cluster_count(self):
        """Recompute dominant color clusters and update the image label accordingly."""
        if self.current_image is None:
            return

        self.color_data.reference_dominant = self.color_data.extract_dominant(
+            self.current_image,
            num_values=15
        )
        dominant_colors = self.color_data.reference_dominant

        # Convert dominant colors to the format expected by the rest of the code
        h, w = self.current_image.shape[:2]
        self.current_labels = np.zeros((h, w), dtype=np.int32)
        self.current_colors = np.array([rgb for (rgb, _) in dominant_colors], dtype=np.uint8)
        
        # Create a mask for each color and assign labels
        self.current_groups = []
        dominant_colors_array = np.array([rgb for rgb, _ in dominant_colors])
        reshaped_image = self.current_image.reshape((-1, 3))
        distances = np.linalg.norm(reshaped_image[:, np.newaxis] - dominant_colors_array, axis=2)
        closest_color_indices = np.argmin(distances, axis=1)
        self.current_labels = closest_color_indices.reshape(self.current_image.shape[:2])
        self.current_groups = np.unique(closest_color_indices).tolist()

        # Update display
        unique_groups = len(set(self.current_groups))

        # Send to display
        self.image_label.setImageData(
            self.current_image,
            self.current_labels,
            self.current_colors,
            self.current_groups
        )

    def update_cluster_info(self, group_idx):
        if self.current_colors is None or self.current_groups is None:
            return

        # Count pixels in this group
        group_mask = np.isin(self.current_groups, [group_idx])
        group_colors = self.current_colors[group_mask]

        # Display group info
        color_info = " | ".join(
            f"RGB({c[0]}, {c[1]}, {c[2]})"
            for c in group_colors
        )

    def display_preview(self, img, is_color):
        """Display the image in the preview label."""
        if img is None:
            return
        
        # Convert BGR to RGB if this is a color image and conversion is requested
        if is_color:
            # Convert BGR to RGB if the image is in BGR format
            if len(img.shape) == 3 and img.shape[2] == 3:  # BGR image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 4:  # BGRA image
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            
            # Convert to QImage/QPixmap based on the image type for display
            if len(img.shape) == 2:  # Grayscale
                height, width = img.shape
                bytes_per_line = width
                qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # RGB/RGBA
                height, width, channels = img.shape
                bytes_per_line = channels * width
                if channels == 3:  # RGB
                    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:  # RGBA
                    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            
            pixmap = QPixmap.fromImage(qimage)
            self.color_left_preview_label.setPixmap(pixmap.scaled(
            self.color_left_preview_label.width(), 
            self.color_left_preview_label.height(), 
            Qt.KeepAspectRatio))
        
        else:
            # Convert to QImage/QPixmap based on the image type
            if len(img.shape) == 2:  # Grayscale
                height, width = img.shape
                bytes_per_line = width
                qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            else:  # RGB/BGR
                height, width, channels = img.shape
                bytes_per_line = channels * width
                if channels == 3:  # RGB/BGR
                    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                else:  # RGBA/BGRA
                    qimage = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            
            pixmap = QPixmap.fromImage(qimage)
            self.value_left_preview_label.setPixmap(pixmap.scaled(self.value_left_preview_label.width(), 
                                                        self.value_left_preview_label.height(), 
                                                        Qt.KeepAspectRatio))

    def upload_image(self, file_path):
        """Load the image, initialize blank/reference canvases for color & value, and display them."""
        # Read image (preserving alpha channel if present)
        self.color_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        self.color_reference_image = self.color_image.copy()
        
        # Create a blank canvas image of the same size
        blank_canvas = np.zeros_like(self.color_image)
        
        # Display with blank canvas on left and reference on right
        self.display_split_view(blank_canvas, self.color_image, True)
        
        # For value analysis - convert to grayscale
        self.value_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if len(self.value_image.shape) == 3:
            if self.value_image.shape[2] == 4:  # RGBA
                self.value_image = cv2.cvtColor(self.value_image, cv2.COLOR_RGBA2GRAY)
            else:  # BGR
                self.value_image = cv2.cvtColor(self.value_image, cv2.COLOR_BGR2GRAY)
        self.value_reference_image = self.value_image.copy()
        
        # Create a blank canvas for value view
        blank_value_canvas = np.zeros_like(self.value_image)
        
        # Display with blank canvas on left and reference on right
        self.display_split_view(blank_value_canvas, self.value_image, False)
            
        # Default to Gaussian filter if none selected
        if not self.current_filter:
            self.gaussian_radio.setChecked(True)
            self.filter_selected("gaussian")
        else:
            self.update_preview()

    def filter_selected(self, filter_type):
        """Set the active filter type, reveal the slider, and refresh the preview."""
        self.current_filter = filter_type
        self.slider_label.show()
        self.slider.show()
        self.update_preview()
    
    def update_kernel_size_label(self, value):
        """Reflect the slider’s value as a percentage in the label text."""
        kernel_percentage = value / 10.0
        self.slider_label.setText(f"Kernel Size (%): {kernel_percentage:.1f}%")

    def update_preview(self):
        """Apply the chosen filter to value and canvas images and update the split view."""
        if self.value_image is None or self.current_filter is None:
            return

        # Make a copy of the image to work with
        working_image = self.value_image.copy()
        
        # Convert to grayscale first
        if len(working_image.shape) == 3:
            if working_image.shape[2] == 4:  # RGBA
                working_image = cv2.cvtColor(working_image, cv2.COLOR_RGBA2GRAY)
            else:  # BGR or RGB
                working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

        # Calculate kernel size
        kernel_percentage = self.slider.value() / 10.0
        hw_max = max(working_image.shape[:2])
        kernel_size = max(3, int(hw_max * (kernel_percentage / 100.0)))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Apply filter to reference image
        if self.current_filter == "gaussian":
            self.filtered_image = cv2.GaussianBlur(working_image, (kernel_size, kernel_size), 0)
        elif self.current_filter == "bilateral":
            sigma_color = 75
            sigma_space = 75
            self.filtered_image = cv2.bilateralFilter(working_image, kernel_size, sigma_color, sigma_space)
        elif self.current_filter == "median":
            self.filtered_image = cv2.medianBlur(working_image, kernel_size)

        # Check if canvas_image has any non-zero values
        if hasattr(self, 'canvas_image') and self.canvas_image is not None and np.any(self.canvas_image):
            self.filtered_canvas = self.canvas_image.copy()
            # Apply the same filter to canvas image
            if self.current_filter == "gaussian":
                self.filtered_canvas = cv2.GaussianBlur(self.filtered_canvas, (kernel_size, kernel_size), 0)
            elif self.current_filter == "bilateral":
                self.filtered_canvas = cv2.bilateralFilter(self.filtered_canvas, kernel_size, sigma_color, sigma_space)
            elif self.current_filter == "median":
                self.filtered_canvas = cv2.medianBlur(self.filtered_canvas, kernel_size)
        else:
            self.filtered_canvas = np.zeros_like(self.filtered_image)
        
        self.display_split_view(self.filtered_canvas, self.filtered_image, False)


    def get_feedback_value(self):
        """Extract dominant values from the canvas and compare with the reference."""
        document = Krita.instance().activeDocument()
        if not document:
            self.value_feedback_label.setText("⚠️ No document is open")
            return

        if self.value_reference_image is None or self.filtered_image is None:
            self.value_feedback_label.setText("⚠️ Please upload a reference image first and apply a filter")
            return
        
        if self.filtered_image is not None:
            # Extract the 20 most dominant values from reference
            self.value_data.reference_dominant = (
                self.value_data.extract_dominant(self.filtered_image, num_values=20)
            )
            self.value_data.create_map_with_blobs(self.filtered_image, use_canvas=False)

        # Get current canvas data
        if hasattr(self, 'filtered_canvas') and self.filtered_canvas is not None:
            pixel_array_gray = self.filtered_canvas
        else:
            pixel_array = self.get_canvas_data()
            if pixel_array is None:
                return

            # Convert pixel array to grayscale
            if pixel_array.shape[2] == 4:  # BGRA format
                pixel_array_gray = cv2.cvtColor(pixel_array, cv2.COLOR_BGRA2GRAY)
            else:
                pixel_array_gray = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2GRAY)

            self.canvas_image = pixel_array_gray

        # Extract dominant values from canvas and reference
        self.value_data.canvas_dominant = (self.value_data.extract_dominant(pixel_array_gray, num_values=5))
        
        # Create value maps and blob information
        self.value_data.create_map_with_blobs(pixel_array_gray, use_canvas=True)
        
        # Match canvas and reference values using spatial information
        self.match_values(is_color_analysis=False)
        
        # Update UI with value pair widgets
        self.update_pairs(
            self.value_data,
            self.value_pairs_container_layout,
            lambda v: v, # no transform for raw grayscale levels
            self.show_pair_regions_value
        )
        
        # Create and display the initial overview with all matched pairs
        self.show_all_matched_pairs(False)
        self.value_feedback_label.setText("✅ Found dominant values. Click on any value pair to see detailed comparison.")
        
    def get_feedback_color(self):
        """Extract dominant colors from the canvas and compare with the reference."""
        document = Krita.instance().activeDocument()
        if not document:
            self.color_feedback_label.setText("⚠️ No document is open")
            return

        if self.color_reference_image is None:
            self.color_feedback_label.setText("⚠️ Please upload a reference image first")
            return
        
        # Get current canvas data
        active_layer = document.activeNode()
        doc_width, doc_height = document.width(), document.height()
        pixel_data = active_layer.pixelData(0, 0, doc_width, doc_height)
        pixel_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(doc_height, doc_width, -1)
        
        # Downsample pixel array to half size using cv2.resize
        pixel_array = cv2.resize(pixel_array, (doc_width//2, doc_height//2), interpolation=cv2.INTER_AREA)

        # Store the original canvas image without format conversion
        self.canvas_image = pixel_array.copy()
        self.filtered_canvas = self.canvas_image.copy()
        
        # Create a version for analysis (RGB)
        analysis_image = pixel_array.copy()
        ref_analysis_image = self.color_reference_image.copy()

        # Extract the 15 most dominant values from reference
        self.color_data.reference_dominant = (
            self.color_data.extract_dominant(ref_analysis_image, num_values=15)
        )
        # Build its reference‐side map & blobs
        self.color_data.create_map_with_blobs(ref_analysis_image, use_canvas=False)

        # Extract the 15 most dominant values from canvas
        self.color_data.canvas_dominant = (
            self.color_data.extract_dominant(analysis_image, num_values=6)
        )
        self.color_data.create_map_with_blobs(analysis_image, use_canvas=True)

        # Create colors maps and blob information
        self.color_data.create_map_with_blobs(ref_analysis_image, use_canvas=False)
        self.color_data.create_map_with_blobs(analysis_image, use_canvas=True)

        # Match canvas and reference colors
        self.match_values(is_color_analysis=True)
        
        # Update UI
        self.update_pairs(
            self.color_data,
            self.color_pairs_container_layout,
            lambda rgb: self.rgb_to_hsv(rgb),  # convert RGB→HSV for display
            self.show_pair_regions_color
        )
        
        # Create and display the initial overview with all matched pairs
        self.show_all_matched_pairs(True)
        text_to_display = "✅ Found dominant colors. Click on any value pair to see detailed comparison."
        self.color_feedback_label.setText(text_to_display)

    def show_all_matched_pairs(self, is_color_analysis=False):
        """Show all matched pairs together - reference with canvas regions and canvas with reference regions."""
        if self.filtered_canvas is None:
            return
        
        # create copies of the reference and canvas images for overlay
        if (not is_color_analysis): 
            if self.filtered_image is None:
                return
            ref_with_regions = cv2.cvtColor(self.filtered_image.copy(), cv2.COLOR_GRAY2BGR)
            canvas_with_regions = cv2.cvtColor(self.filtered_canvas.copy(), cv2.COLOR_GRAY2BGR)
            matched_pairs = self.value_data.matched_pairs
        else:
            if self.color_reference_image is None:
                return
            ref_with_regions = self.color_reference_image.copy()
            canvas_with_regions = self.filtered_canvas.copy()
            matched_pairs = self.color_data.matched_pairs

        # Store 5 colors for getting all matched pairs
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (128, 0, 128)]
        
        # Draw all matched pairs
        i = 0
        for canvas_hex, ref_hex in matched_pairs.items():
            #  region masks
            cur_color = colors[i % len(colors)]
            canvas_mask = self.get_region_mask(self.filtered_canvas, canvas_hex, is_color_analysis)
            if (is_color_analysis):
                ref_mask = self.get_region_mask(self.color_reference_image, ref_hex, is_color_analysis)
            else:
                ref_mask = self.get_region_mask(self.filtered_image, ref_hex, is_color_analysis)

            #  Contours for both masks
            canvas_contours, _ = cv2.findContours(canvas_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ref_contours, _ = cv2.findContours(ref_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if (is_color_analysis):
                # Filter contours by minimum area
                min_area = 100  # Adjust this threshold as needed
                canvas_contours = [c for c in canvas_contours if cv2.contourArea(c) > min_area]
                ref_contours = [c for c in ref_contours if cv2.contourArea(c) > min_area]
            
            for contour in canvas_contours:
                if len(contour) > 2:
                    smoothed_contour = self.smooth_contour(contour)
                    cv2.polylines(canvas_with_regions, [smoothed_contour], 
                                isClosed=True, color=cur_color, thickness=9)
            
            for contour in ref_contours:
                if len(contour) > 2:
                    smoothed_contour = self.smooth_contour(contour)
                    cv2.polylines(ref_with_regions, [smoothed_contour], 
                                    isClosed=True, color=cur_color, thickness=9)
            
            i+=1
            
        # Display the overlays - canvas on left, reference on right
        self.display_split_view(canvas_with_regions, ref_with_regions, is_color_analysis)

    def display_split_view(self, left_image, right_image, is_color_analysis=False):
        """Display two images side by side in the preview labels.
        
        Args:
            left_image: First image to display (numpy array) - or canvas
            right_image: Second image to display (numpy array) - or reference
            is_color: Whether to handle as color images (True) or values (False)
        """
        # Create copies to avoid modifying originals
        left_display = left_image.copy()
        right_display = right_image.copy()

        # Convert images to display format
        if is_color_analysis:
            # Color display conversions
            if len(left_display.shape) == 2:  # Grayscale
                left_display = cv2.cvtColor(left_display, cv2.COLOR_GRAY2RGB)
            elif left_display.shape[2] == 4:  # RGBA
                left_display = cv2.cvtColor(left_display, cv2.COLOR_BGRA2RGB)
            elif left_display.shape[2] == 3:  # BGR
                left_display = cv2.cvtColor(left_display, cv2.COLOR_BGR2RGB)
            
            if len(right_display.shape) == 2:  # Grayscale
                right_display = cv2.cvtColor(right_display, cv2.COLOR_GRAY2RGB)
            elif right_display.shape[2] == 4:  # RGBA
                right_display = cv2.cvtColor(right_display, cv2.COLOR_BGRA2RGB)
            elif right_display.shape[2] == 3:  # BGR
                right_display = cv2.cvtColor(right_display, cv2.COLOR_BGR2RGB)
        else:
            # Value display - ensure grayscale is shown as RGB for consistent display
            if len(left_display.shape) == 2:  # Grayscale
                left_display = cv2.cvtColor(left_display, cv2.COLOR_GRAY2RGB)
            elif left_display.shape[2] == 4:  # RGBA
                left_display = cv2.cvtColor(left_display, cv2.COLOR_BGRA2RGB)
            elif left_display.shape[2] == 3:  # BGR
                left_display = cv2.cvtColor(left_display, cv2.COLOR_BGR2RGB)
                
            if len(right_display.shape) == 2:  # Grayscale
                right_display = cv2.cvtColor(right_display, cv2.COLOR_GRAY2RGB)
            elif right_display.shape[2] == 4:  # RGBA
                right_display = cv2.cvtColor(right_display, cv2.COLOR_BGRA2RGB)
            elif right_display.shape[2] == 3:  # BGR
                right_display = cv2.cvtColor(right_display, cv2.COLOR_BGR2RGB)

        left_label = self.color_left_preview_label if is_color_analysis else self.value_left_preview_label
        right_label = self.color_right_preview_label if is_color_analysis else self.value_right_preview_label

        # Convert left and right image 
        left_height, left_width = left_display.shape[:2]
        left_bytes_per_line = left_display.shape[2] * left_width
        left_qimage = QImage(left_display.data, left_width, left_height, 
                            left_bytes_per_line, QImage.Format_RGB888)
        left_pixmap = QPixmap.fromImage(left_qimage)
        
        right_height, right_width = right_display.shape[:2]
        right_bytes_per_line = right_display.shape[2] * right_width
        right_qimage = QImage(right_display.data, right_width, right_height, 
                            right_bytes_per_line, QImage.Format_RGB888)
        right_pixmap = QPixmap.fromImage(right_qimage)
        
        # Update the preview labels
        left_label.setPixmap(left_pixmap.scaled(
            left_label.width(), 
            left_label.height(), 
            Qt.KeepAspectRatio))
        
        right_label.setPixmap(right_pixmap.scaled(
            right_label.width(), 
            right_label.height(), 
            Qt.KeepAspectRatio))
        
    def smooth_contour(self, contour, num_points=100):
        """Smooth a contour using spline interpolation."""
        # Extract points from the contour
        points = contour.reshape(-1, 2)
        if len(points) < 3:  # Need at least 3 points for smoothing
            return contour
            
        x = points[:, 0]
        y = points[:, 1]

        # Add the first point to the end to close the loop
        x = np.append(x, x[0])
        y = np.append(y, y[0])

        # Create a cumulative distance array for interpolation
        t = np.zeros(len(x))
        t[1:] = np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2)
        t = np.cumsum(t)
        
        if t[-1] == 0:
            return contour
        t /= t[-1]

        # Generate new points using spline interpolation
        t_new = np.linspace(0, 1, num_points)
        x_new = np.interp(t_new, t, x)
        y_new = np.interp(t_new, t, y)

        # Combine into a list of points
        smoothed_points = np.array([x_new, y_new]).T.astype(np.int32)
        return smoothed_points.reshape(-1, 1, 2)  # Reshape for polylines function
    
    def calculate_bbox_overlap(self, bbox1, bbox2):
        """Calculate the overlap ratio between two bounding boxes."""
        if not bbox1 or not bbox2:
            return 0.0
                
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate coordinates of the intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            # No overlap
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU 
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def calculate_color_similarity(self, hex1, hex2, is_color_analysis=False):
        """
        Calculate value‐similarity S_val between two hex colors per:
        S_val = 1 - (1/3) * ||ΔLab|| after normalizing L*∈[0,1], a*∈[0,1], b*∈[0,1].
        """

        def hex_to_lab(hex_color):
            # 1) Hex → sRGB [0–1]
            r = int(hex_color[1:3], 16) / 255.0
            g = int(hex_color[3:5], 16) / 255.0
            b = int(hex_color[5:7], 16) / 255.0

            def inv_gamma(c):
                return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
            r_lin, g_lin, b_lin = inv_gamma(r), inv_gamma(g), inv_gamma(b)

            # 2) Linear RGB → XYZ (D65)
            x = (r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375) * 100
            y = (r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750) * 100
            z = (r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041) * 100

            # 3) Normalize by D65 white point
            x, y, z = x / 95.047, y / 100.000, z / 108.883

            # 4) XYZ → Lab
            def f(t):
                return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)
            fx, fy, fz = f(x), f(y), f(z)
            L = 116 * fy - 16
            a = 500 * (fx - fy)
            b = 200 * (fy - fz)
            return L, a, b

        # Convert both hex colors to Lab
        L1, a1, b1 = hex_to_lab(hex1)
        L2, a2, b2 = hex_to_lab(hex2)

        Ln1, an1, bn1 = L1 / 100.0, (a1 + 128) / 255.0, (b1 + 128) / 255.0
        Ln2, an2, bn2 = L2 / 100.0, (a2 + 128) / 255.0, (b2 + 128) / 255.0

        # Euclidean distance in the normalized cube (max = √3)
        delta = np.sqrt(
            (Ln1 - Ln2)**2 +
            (an1 - an2)**2 +
            (bn1 - bn2)**2
        )

        # Formula: S_val = 1 – (1/3) * Δ
        similarity = 1.0 - (delta / 3.0)

        # Clamp to [0,1]
        return max(0.0, min(1.0, similarity))
        
    def match_values(self, is_color_analysis=False):
        """
        Match canvas values to reference values based on spatial and color information.
        Only consider values that have actual blob regions on the screen.
        
        Args:
            is_color_analysis: Whether to match colors (True) or values (False)
        """
        if is_color_analysis:
            self.color_data.matched_pairs = self.match_values_generic(
                self.color_data.canvas_dominant,
                self.color_data.reference_dominant,
                self.color_data.canvas_blobs,
                self.color_data.reference_blobs,
                True  # is_color_analysis
            )
        else:
            self.value_data.matched_pairs = self.match_values_generic(
                self.value_data.canvas_dominant,
                self.value_data.reference_dominant,
                self.value_data.canvas_blobs,
                self.value_data.reference_blobs,
                False  # is_color_analysis
            )

    def match_values_generic(self, canvas_values, reference_values, canvas_blobs, reference_blobs, is_color_analysis):
        """
        Generic function to match canvas values to reference values based on spatial and color information.
        Only consider values that have actual blob regions on the screen.
        
        Args:
            canvas_values: List of (value/rgb, hex_code) tuples for canvas
            reference_values: List of (value/rgb, hex_code) tuples for reference
            canvas_blobs: Dictionary of blob information for canvas
            reference_blobs: Dictionary of blob information for reference
            is_color_analysis: Whether this is color analysis (True) or value analysis (False)
            
        Returns:
            Dictionary of matched pairs {canvas_hex: reference_hex}
        """
        matched_pairs = {}
        
        canvas_values_with_blobs = []
        for (value, hex_code) in canvas_values:
            blob = canvas_blobs.get(hex_code)   
            if blob and blob.points:                            
                canvas_values_with_blobs.append((value, hex_code))

        reference_values_with_blobs = []
        for (value, hex_code) in reference_values:
            blob = reference_blobs.get(hex_code)   
            if blob and blob.points:                            
                reference_values_with_blobs.append((value, hex_code))
        
        if not canvas_values_with_blobs or not reference_values_with_blobs:
            return matched_pairs
        
        # Calculate similarity matrix between all canvas and reference values with blobs
        similarity_matrix = []
        
        for c_value, c_hex in canvas_values_with_blobs:
            c_bbox = canvas_blobs[c_hex].bbox
            row = []
            for r_value, r_hex in reference_values_with_blobs:
                r_bbox = reference_blobs[r_hex].bbox

                # Calculate color similarity (weighted 40%)
                color_similarity = self.calculate_color_similarity(c_hex, r_hex, is_color_analysis) * 0.50
                
                # Calculate spatial similarity (weighted 60%)
                spatial_similarity = self.calculate_bbox_overlap(c_bbox, r_bbox) * 0.50
                
                # Total similarity is weighted sum
                total_similarity = color_similarity + spatial_similarity
                row.append((total_similarity, r_hex))
            
            similarity_matrix.append((c_hex, row))
        
        # Greedy matching algorithm for canvas to reference
        for c_hex, similarities in similarity_matrix:
            best_match = max(similarities, key=lambda x: x[0])
            best_similarity, best_ref_hex = best_match
            
            # Only match if similarity is above threshold
            if best_similarity >= 0.01:  
                matched_pairs[c_hex] = best_ref_hex
        
        return matched_pairs

    def rgb_to_hsv(self, rgb):
        """
        Convert RGB tuple to HSV tuple using OpenCV, 
        converting to standard display ranges.
        
        Args:
            rgb (tuple): RGB color values (0-255 range)
        
        Returns:
            tuple: HSV color values in standard ranges
                Hue: 0-360
                Saturation: 0-100
                Value: 0-100
        """
        # Create a single pixel numpy array in RGB format
        rgb_pixel = np.uint8([[[rgb[0], rgb[1], rgb[2]]]]) 
        
        # Convert RGB to HSV
        hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)
        
        # Extract OpenCV HSV values
        h, s, v = hsv_pixel[0, 0]
        
        # Convert to standard ranges
        # Hue: 0-179 -> 0-360
        h_standard = h * 2
        
        # Saturation: 0-255 -> 0-100
        s_standard = round(s / 255 * 100)
        
        # Value: 0-255 -> 0-100
        v_standard = round(v / 255 * 100)
        
        return (int(h_standard), int(s_standard), int(v_standard))
    
    def update_pairs(self, data, container_layout, transform, click_fn):
        """
        Generic UI refresher for both value and color pairs.

        - data: either self.value_data or self.color_data
        - container_layout: either self.value_pairs_container_layout or self.color_pairs_container_layout
        - transform: a function(feature) → display_value  (e.g. identity or rgb→hsv)
        - click_fn: either self.show_pair_regions_value or self.show_pair_regions_color
        """
        # Clear existing widgets
        #    They are in either self.value_pair_widgets or self.color_pair_widgets,
        #    Iterate container_layout’s children
        while container_layout.count():
            w = container_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

        # Re-build widgets
        for canvas_hex, ref_hex in data.matched_pairs.items():
            left_feat  = next(v for v, h in data.canvas_dominant    if h == canvas_hex)
            right_feat = next(v for v, h in data.reference_dominant if h == ref_hex)

            # transform it for display (e.g. value→value or rgb→hsv)
            left_disp  = transform(left_feat)
            right_disp = transform(right_feat)

            pair = ValuePairWidget(left_disp, canvas_hex, right_disp, ref_hex)
            pair.canvas_button.clicked.connect(lambda _, c=canvas_hex, r=ref_hex: click_fn(c, r))
            pair.ref_button   .clicked.connect(lambda _, c=canvas_hex, r=ref_hex: click_fn(c, r))
            # add to the UI
            container_layout.addWidget(pair)

    def get_region_mask(self, image, hex_code, is_color_analysis=False):
        """Get a binary mask for a specific value region."""
        if (is_color_analysis):
            """Get a binary mask for a specific RGB value region."""
            # Convert hex code to RGB values
            r = int(hex_code[1:3], 16)
            g = int(hex_code[3:5], 16)
            b = int(hex_code[5:7], 16)
            
            # Define threshold for RGB similarity
            threshold = 20
            
            # Create a temporary RGB copy for masking but don't modify original
            # This is beacuse cv2 requires RGB for masking
            rgb_image = image.copy()
            if len(rgb_image.shape) == 2:  # If grayscale, convert to RGB
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
            elif rgb_image.shape[2] == 4:  # If RGBA, convert to RGB
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGB)
            elif rgb_image.shape[2] == 3 and image is not self.canvas_image:  # If BGR and not canvas, convert to RGB
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                
            # Create lower and upper bounds for each channel
            lower_bound = np.array([
                max(0, r - threshold),
                max(0, g - threshold),
                max(0, b - threshold)
            ])
            upper_bound = np.array([
                min(255, r + threshold),
                min(255, g + threshold),
                min(255, b + threshold)
            ])
            
            # Create a binary mask where the pixel values are within the threshold range
            mask = cv2.inRange(rgb_image, lower_bound, upper_bound)
        else:
            value = int(hex_code[1:3], 16)  # Extract grayscale value from hex
            threshold = 15  # Threshold for value similarity
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # BGRA
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
                else:  # BGR
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
            
            mask = np.zeros_like(gray_image, dtype=np.uint8)
            lower_bound = max(0, value - threshold)
            upper_bound = min(255, value + threshold)
            mask[(gray_image >= lower_bound) & (gray_image <= upper_bound)] = 255
        
        return mask
    
    hue_ranges = [
        (0, 30, "red"),
        (30, 90, "yellow"),
        (90, 150, "green"),
        (150, 210, "cyan"),
        (210, 270, "blue"),
        (270, 330, "magenta")
    ]

    def get_significant_contours(self, mask: np.ndarray, min_area: int = 150):
        """Return only those contours in `mask` whose area > min_area."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > min_area]

    def draw_contours(self, img: np.ndarray, contours: list, color: tuple, thickness: int = 9):
        """Draw smoothed contours onto `img` in-place."""
        for cnt in contours:
            if len(cnt) > 2:
                sm = self.smooth_contour(cnt)
                cv2.polylines(img, [sm], isClosed=True, color=color, thickness=thickness)

    def overlay_and_show(self,
                          canvas_img: np.ndarray,
                          ref_img:    np.ndarray,
                          canvas_hex: str,
                          ref_hex:    str,
                          is_color:   bool):
        """
        1) mask → contours
        2) draw red on canvas, green on reference
        3) display split
        4) if color, save out overlays for Krita
        """
        # Build masks
        m_c = self.get_region_mask(canvas_img, canvas_hex, is_color)
        m_r = self.get_region_mask(ref_img,    ref_hex,    is_color)

        # Pick only the big ones
        ct_c = self.get_significant_contours(m_c)
        ct_r = self.get_significant_contours(m_r)

        cv = (cv2.cvtColor(canvas_img, cv2.COLOR_GRAY2BGR)
              if (not is_color and canvas_img.ndim == 2)
              else canvas_img.copy())
        rv = (cv2.cvtColor(ref_img,    cv2.COLOR_GRAY2BGR)
              if (not is_color and ref_img.ndim == 2)
              else ref_img.copy())

        # Draw red / green
        self.draw_contours(cv, ct_c, (0, 0, 255))
        self.draw_contours(rv, ct_r, (0, 255, 0))

        self.display_split_view(cv, rv, is_color)

        # Save for Krita (only for color analysis)
        if is_color:
            temp_dir = os.path.join(
                os.path.expanduser("~/Library/Application Support/krita/pykrita/artkrit")
                if sys.platform=='darwin'
                else os.path.expanduser("~/.local/share/krita/pykrita/artkrit"),
                "temp"
            )
            os.makedirs(temp_dir, exist_ok=True)
            cv2.imwrite(os.path.join(temp_dir, "canvas_color_overlay.png"),   cv)
            cv2.imwrite(os.path.join(temp_dir, "reference_color_overlay.png"), rv)

    def compute_hue_feedback(self, canvas_hsv, reference_hsv):
        """Given two HSV triples (hue, saturation, value), return a string
        explaining how your canvas hue/saturation compares to the reference."""
        # Unpack the HSV components for clarity
        canvas_hue, canvas_saturation, canvas_value = canvas_hsv
        ref_hue, ref_saturation, ref_value = reference_hsv

        # Helper: map a hue angle to its descriptive range name
        def _hue_label(angle):
            for start, end, name in self.hue_ranges:
                if start <= angle < end:
                    return name
            return "unknown"

        # Find which named bucket each hue falls into
        canvas_label = _hue_label(canvas_hue)
        ref_label = _hue_label(ref_hue)

        hue_diff = canvas_hue - ref_hue
        if ref_saturation:
            sat_diff = (canvas_saturation - ref_saturation) / ref_saturation * 100
        else:
            sat_diff = 0

        if canvas_label == ref_label:
            if hue_diff == 0:
                hue_feedback = "Your canvas matches the reference exactly in hue."
            else:
                direction = "warmer" if hue_diff > 0 else "cooler"
                intensity = "slightly" if abs(hue_diff) < 10 else "quite"
                hue_feedback = (
                    f"You're in the same {canvas_label} range as the reference, "
                    f"but the reference is {intensity} {direction}. "
                    f"Consider adjusting to match more closely."
                )
        else:
            direction = "warmer" if hue_diff > 0 else "cooler"
            hue_feedback = (
                f"Your canvas sits in the {canvas_label} range, "
                f"while the reference sits in the {ref_label} range. "
                f"You should move {direction} towards {ref_label}."
            )

        if sat_diff > 5:
            sat_feedback = f"Your color is {abs(sat_diff):.1f}% more saturated than the reference."
        elif sat_diff < -5:
            sat_feedback = f"Your color is {abs(sat_diff):.1f}% less saturated than the reference."
        else:
            sat_feedback = "Your color has similar saturation to the reference."

        feedback = "HSV Differences:\n"
        feedback += f"Hue Difference: {hue_diff}°\n"
        feedback += hue_feedback + "\n"
        feedback += sat_feedback + "\n"

        return feedback

    def get_color_feedback(self, canvas_hsv, reference_hsv):
        """Generate feedback about the color comparison."""
        feedback = self.compute_hue_feedback(canvas_hsv, reference_hsv)
        return feedback 

    def get_value_feedback(self, canvas_value, ref_value):
        """Generate feedback about the value comparison."""
        # Calculate the difference between canvas and reference values
        canvas_value = int(canvas_value)
        ref_value = int(ref_value)
        value_diff = canvas_value - ref_value
        
        feedback = ""
        
        # Set thresholds for differences
        minor_threshold = 3
        moderate_threshold = 20
        major_threshold = 30
        
        if abs(value_diff) <= minor_threshold:
            feedback = f"These values are closely matched (difference: {abs(value_diff)})"
        else:
            if value_diff < 0:
                # Canvas is darker than reference
                if abs(value_diff) > major_threshold:
                    feedback = f"Canvas values are significantly too dark (by {abs(value_diff)} levels)"
                elif abs(value_diff) > moderate_threshold:
                    feedback = f"Canvas values are moderately too dark (by {abs(value_diff)} levels)"
                else:
                    feedback = f"Canvas values are slightly too dark (by {abs(value_diff)} levels)"
            else:
                # Canvas is lighter than reference
                if abs(value_diff) > major_threshold:
                    feedback = f"Canvas values are significantly too light (by {abs(value_diff)} levels)"
                elif abs(value_diff) > moderate_threshold:
                    feedback = f"Canvas values are moderately too light (by {abs(value_diff)} levels)"
                else:
                    feedback = f"Canvas values are slightly too light (by {abs(value_diff)} levels)"
        
        if abs(value_diff) > minor_threshold:
            if value_diff < 0:
                feedback += "\nSuggestion: Try lightening this area to better match the reference."
            else:
                feedback += "\nSuggestion: Try darkening this area to better match the reference."
        
        return feedback
    
    def show_pair_regions_color(self, canvas_hex, ref_hex):
        """Highlight the selected color pair on canvas and reference, and show feedback."""
        if self.color_reference_image is None or self.canvas_image is None:
            return

        # look up the two RGBs, bail if missing
        c_rgb = next((rgb for rgb,h in self.color_data.canvas_dominant    if h==canvas_hex), None)
        r_rgb = next((rgb for rgb,h in self.color_data.reference_dominant if h==ref_hex),    None)
        if not (c_rgb and r_rgb):
            return

        # feedback label
        feedback = self.get_color_feedback(self.rgb_to_hsv(c_rgb), self.rgb_to_hsv(r_rgb))
        self.color_feedback_label.setText(feedback)

        # overlay & show
        self.overlay_and_show(self.canvas_image, self.color_reference_image,
                               canvas_hex, ref_hex, is_color=True)

    def show_pair_regions_value(self, canvas_hex, ref_hex):
        """Highlight the selected value pair on canvas and reference, and show feedback."""
        if self.filtered_canvas is None or self.filtered_image is None:
            return

        # feedback text (reuse existing get_value_feedback)
        c_val = next((v for v,h in self.value_data.canvas_dominant    if h==canvas_hex), None)
        r_val = next((v for v,h in self.value_data.reference_dominant if h==ref_hex),    None)
        if c_val is not None and r_val is not None:
            self.value_feedback_label.setText(self.get_value_feedback(c_val, r_val))

        # do exactly the same overlay→show, but in grayscale mode
        self.overlay_and_show(self.filtered_canvas, self.filtered_image,
                               canvas_hex, ref_hex, is_color=False)
        
    def canvasChanged(self, event=None):
        """This method is called whenever the canvas changes. It's required for all DockWidget subclasses in Krita."""
        pass
    
    def selectColor(self):
        """Open the HS color picker and apply the chosen color to Krita’s foreground."""
        dialog = CustomHSColorPickerDialog(self)
        dialog.exec_()
        
        selectedColor = dialog.selectedColor()
        if selectedColor.isValid():
            print(f"Selected Color: {selectedColor.name()}")
            self.colorButton.setText(f"Color: {selectedColor.name()}")
            
            if Krita.instance().activeWindow() and Krita.instance().activeWindow().activeView():
                view = Krita.instance().activeWindow().activeView()
                managedColor = ManagedColor("RGBA", "U8", "")
                managedColor.setComponents([
                    selectedColor.blueF(),
                    selectedColor.greenF(),
                    selectedColor.redF(),
                    1.0
                ])
                view.setForeGroundColor(managedColor)

    def activateLassoTool(self):
        """Activate Krita’s lasso selection tool and prepare the fill options UI."""
        krita_instance = Krita.instance()
        
        action = krita_instance.action('KisToolSelectContiguous')
        if action:
            action.trigger()
            self.lassoButton.setStyleSheet("background-color: #AED6F1;")
            
            self.fillGroup.setVisible(True)
            
            self.selectionTimer.start(500)
            
            QTimer.singleShot(500, lambda: self.lassoButton.setStyleSheet(""))
    
    def selectFillColor(self):
        """Open the HS picker seeded by the current selection’s average value."""
        # Extract the dominant value from the selection
        doc = Krita.instance().activeDocument()
        if doc:
            selection = doc.selection()
            if selection:
                node = doc.activeNode()
                average_value = self.extractAverageValueFromSelection(node, selection)
                if average_value is not None:
                    dialog = CustomHSColorPickerDialog(self, average_value)
                    dialog.exec_()
                    
                    selectedColor = dialog.selectedColor()
                    if selectedColor.isValid():
                        self.currentFillColor = selectedColor
                        self.fillColorButton.setStyleSheet(f"background-color: {selectedColor.name()};")

    def fillSelection(self):
        """Fill the current selection with the selected fill color."""
        krita_instance = Krita.instance()
        doc = krita_instance.activeDocument()
        selection = doc.selection()
        node = doc.activeNode()
        average_value = self.extractAverageValueFromSelection(node, selection)
        if average_value is None:
            print("Failed to extract average value from selection")
            return
            
        # Get the selected H and S from the color picker
        selected_hue = self.currentFillColor.hue()
        selected_saturation = self.currentFillColor.saturation()
        
        # Create the new color with the extracted value and selected H and S
        new_color = QColor.fromHsv(selected_hue, selected_saturation, average_value)
        print(f"New fill color: {new_color.name()}")
        
        # Convert QColor to ManagedColor for Krita
        managedColor = ManagedColor("RGBA", "U8", "")
        managedColor.setComponents([
            new_color.blueF(),
            new_color.greenF(),
            new_color.redF(),
            1.0  # Fully opaque
        ])
        
        # Set the foreground color
        if krita_instance.activeWindow() and krita_instance.activeWindow().activeView():
            view = krita_instance.activeWindow().activeView()
            view.setForeGroundColor(managedColor)
            print("Foreground color set to new color")
            
            # Trigger the fill tool action
            fillToolAction = krita_instance.action('KritaFill/KisToolFill')
            if fillToolAction:
                print("Triggering fill tool action")
                fillToolAction.trigger()
                
                QTimer.singleShot(100, lambda: self.triggerFillForeground(krita_instance))
            else:
                print("Could not find fill tool action")

    def triggerFillForeground(self, krita_instance):
        """
        Triggers the fill_foreground action after the fill tool is activated.
        """
        fillAction = krita_instance.action('fill_foreground')
        if fillAction:
            fillAction.trigger()

    def extractAverageValueFromSelection(self, node, selection):
        """
        Extracts the dominant brightness (value) from the selected area using the HSV color space.
        """
        try:
            print("Extracting pixel data from selection...")
            
            # Get the pixel data from the selected area
            pixel_data = node.projectionPixelData(
                selection.x(), selection.y(), selection.width(), selection.height()
            ).data()
            
            pixels = []
            for i in range(0, len(pixel_data), 4):
                r = pixel_data[i]
                g = pixel_data[i + 1]
                b = pixel_data[i + 2]
                pixels.append((r, g, b))
            
            # Calculate the frequency of each brightness (value) level
            value_counts = {}
            for r, g, b in pixels:
                # Convert RGB to HSV
                hsv_color = QColor(r, g, b).getHsv()
                value = hsv_color[2]
                
                # Count the frequency of each value
                if value in value_counts:
                    value_counts[value] += 1
                else:
                    value_counts[value] = 1
            
            # Find the dominant value (the one with the highest frequency)
            dominant_value = max(value_counts, key=value_counts.get)
            print(f"Dominant value (brightness): {dominant_value}")
            
            return dominant_value
        
        except Exception as e:
            print(f"Error extracting dominant value: {str(e)}")
            return None
    
    def checkSelection(self):
        doc = Krita.instance().activeDocument()
        if doc:
            selection = doc.selection()
            if selection:
                self.fillButton.setEnabled(True)
            else:
                self.fillButton.setEnabled(False)
                
        if self.fillGroup.isVisible():
            self.selectionTimer.start(500)

    def zoom_in(self):
        """Zoom in on the image"""
        self.image_label.setMinimumSize(
            int(self.image_label.minimumWidth() * 1.2),
            int(self.image_label.minimumHeight() * 1.2)
        )
        self.image_label.setMaximumSize(
            int(self.image_label.maximumWidth() * 1.2),
            int(self.image_label.maximumHeight() * 1.2)
        )
        self.image_label.update()

    def zoom_out(self):
        """Zoom out on the image"""
        self.image_label.setMinimumSize(
            int(self.image_label.minimumWidth() / 1.2),
            int(self.image_label.minimumHeight() / 1.2)
        )
        self.image_label.setMaximumSize(
            int(self.image_label.maximumWidth() / 1.2),
            int(self.image_label.maximumHeight() / 1.2)
        )
        self.image_label.update()

    def get_canvas_data(self):
        """Get the current canvas data as a numpy array."""
        document = Krita.instance().activeDocument()
        if not document:
            return None

        active_layer = document.activeNode()
        doc_width, doc_height = document.width(), document.height()
        pixel_data = active_layer.pixelData(0, 0, doc_width, doc_height)
        pixel_array = np.frombuffer(pixel_data, dtype=np.uint8).reshape(doc_height, doc_width, -1)
        
        # Downsample pixel array to half size using cv2.resize
        pixel_array = cv2.resize(pixel_array, (doc_width//2, doc_height//2), interpolation=cv2.INTER_AREA)
        
        return pixel_array

    def show_current_canvas(self):
        """Show the current canvas data in grayscale in the left preview."""
        pixel_array = self.get_canvas_data()
        if pixel_array is None:
            self.value_feedback_label.setText("⚠️ No document is open")
            return

        # Convert to grayscale
        if pixel_array.shape[2] == 4:  # BGRA format
            pixel_array_gray = cv2.cvtColor(pixel_array, cv2.COLOR_BGRA2GRAY)
        else:
            pixel_array_gray = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2GRAY)

        self.canvas_image = pixel_array_gray

        # Display the grayscale image
        self.display_preview(pixel_array_gray, False)
        self.value_feedback_label.setText("✅ Showing current canvas in grayscale")


class CustomHSColorPickerDialog(QDialog):
    def __init__(self, parent=None, extracted_value=None):
        super().__init__(parent)
        
        self.setWindowTitle("Select Color")
        self.setModal(True)
        
        self.currentColor = QColor(255, 0, 0)
        self.currentHue = 0
        self.currentSaturation = 255
        self.currentValue = extracted_value  # Use the extracted value (if provided)
        
        self.huePicker = HuePicker(self)
        self.saturationValuePicker = SaturationValuePicker(self, extracted_value)
        
        self.colorPreview = QLabel()
        self.colorPreview.setFixedSize(100, 100)
        self.colorPreview.setStyleSheet(f"background-color: {self.currentColor.name()};")
        
        self.okButton = QPushButton("OK")
        self.okButton.clicked.connect(self.accept)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.reject)
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select Hue"))
        layout.addWidget(self.huePicker)
        layout.addWidget(QLabel("Select Saturation"))
        layout.addWidget(self.saturationValuePicker)
        layout.addWidget(self.colorPreview)
        layout.addWidget(self.okButton)
        layout.addWidget(self.cancelButton)

        self.setLayout(layout)
        
        self.huePicker.colorChanged.connect(self.updateFromHue)
        self.saturationValuePicker.colorChanged.connect(self.updateColor)

    def updateFromHue(self):
        self.currentHue = self.huePicker.getHue()
        self.saturationValuePicker.setHue(self.currentHue)
        self.updateColor()

    def updateColor(self):
        self.currentColor = self.saturationValuePicker.getColor()
        self.colorPreview.setStyleSheet(f"background-color: {self.currentColor.name()};")

    def selectedColor(self):
        return self.currentColor

class HuePicker(QWidget):
    colorChanged = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.setStyleSheet("background-color: #f1f1f1; border: 1px solid #ccc;")
        
        self.hue = 0
        self.setAutoFillBackground(True)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        
        outer_radius = min(rect.width(), rect.height()) / 2
        inner_radius = outer_radius - 20
        
        gradient = QConicalGradient(rect.center(), 90)
        for i in range(360):
            gradient.setColorAt(i / 360, QColor.fromHsv(i, 255, 255))
        
        path = QPainterPath()
        path.addEllipse(rect.center(), outer_radius, outer_radius)
        path.addEllipse(rect.center(), inner_radius, inner_radius)
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.updateHue(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.updateHue(event.pos())
    
    def updateHue(self, pos):
        center = self.rect().center()
        dx = pos.x() - center.x()
        dy = pos.y() - center.y()
        angle = math.degrees(math.atan2(dy, dx))
        self.hue = int((360 - ((angle + 90 + 360) % 360)) % 360)
        
        self.update()
        self.colorChanged.emit()

    def getHue(self):
        return self.hue

class SaturationValuePicker(QWidget):
    colorChanged = pyqtSignal()

    def __init__(self, parent=None, extracted_value=None):
        super().__init__(parent)
        self.setFixedSize(200, 50 if extracted_value is not None else 200)
        self.setStyleSheet("background-color: #f1f1f1; border: 1px solid #ccc;")
        
        self.hue = 0
        self.saturation = 255
        self.value = extracted_value  # Use the extracted value (if provided)
        self.extracted_value = extracted_value
        self.setAutoFillBackground(True)
        
    def setHue(self, hue):
        """
        Set the hue for the color range.
        """
        self.hue = hue
        self.update()

    def paintEvent(self, event):
        """
        Draw either a full saturation-value square or a horizontal slider for saturation.
        """
        painter = QPainter(self)
        rect = self.rect()
        
        if self.extracted_value is None:
            # Draw the full saturation-value square
            for x in range(rect.width()):
                for y in range(rect.height()):
                    saturation = int((x / rect.width()) * 255)
                    value = int((y / rect.height()) * 255)
                    color = QColor.fromHsv(self.hue, saturation, value)
                    painter.setPen(color)
                    painter.drawPoint(x, y)
        else:
            # Draw a horizontal slider for saturation (value is fixed)
            for x in range(rect.width()):
                saturation = int((x / rect.width()) * 255)
                color = QColor.fromHsv(self.hue, saturation, self.extracted_value)
                painter.setPen(color)
                painter.drawLine(x, 0, x, rect.height())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.updateColorFromPosition(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.updateColorFromPosition(event.pos())

    def updateColorFromPosition(self, pos):
        """
        Update the selected color based on the mouse position.
        """
        rect = self.rect()
        x = pos.x()
        
        # Calculate the saturation based on the x position
        self.saturation = int((x / rect.width()) * 255)
        
        # If no extracted value, calculate the value based on the y position
        if self.extracted_value is None:
            y = pos.y()
            self.value = int((y / rect.height()) * 255)
        
        # Emit the color change signal
        self.colorChanged.emit()

    def getColor(self):
        """
        Get the currently selected color.
        """
        return QColor.fromHsv(self.hue, self.saturation, self.value)
    
    
class ClusterHoverLabel(QLabel):
    clusterHovered = pyqtSignal(int)  # Signal when cluster is hovered

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.original_img = None
        self.cluster_labels = None
        self.dominant_colors = None
        self.cluster_groups = None
        self.current_pixmap = None
        self._pixmap_offset = QPoint(0, 0)  # Track pixmap position within label
        self._pixmap_scale = 1.0  # Track scaling factor

    def setImageData(self, original_img, labels, colors, groups):
        self.original_img = original_img
        self.cluster_labels = labels
        self.dominant_colors = colors
        self.cluster_groups = groups

        h, w = original_img.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(original_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(self.scalePixmap(self.current_pixmap))

    def scalePixmap(self, pixmap):
        if pixmap.isNull():
            return pixmap

        # Get available size from the parent scroll area
        available_width = max(100, self.parent().width() - 20)
        available_height = self.parent().height() - 20

        # Calculate scaled size maintaining aspect ratio
        pixmap_ratio = pixmap.width() / pixmap.height()
        available_ratio = available_width / available_height

        if pixmap_ratio > available_ratio:
            scaled_width = available_width
            scaled_height = int(scaled_width / pixmap_ratio)
        else:
            scaled_height = available_height
            scaled_width = int(scaled_height * pixmap_ratio)

        # Store scaling factor and offset for accurate coordinate mapping
        self._pixmap_scale = scaled_width / pixmap.width()

        # Calculate the centered position
        self._pixmap_offset = QPoint(
            (self.width() - scaled_width) // 2,
            (self.height() - scaled_height) // 2
        )

        return pixmap.scaled(
            scaled_width, scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

    def mouseMoveEvent(self, event):
        if self.original_img is None or self.current_pixmap is None:
            return

        # Convert mouse position to pixmap coordinates
        pos = event.pos()

        # Check if mouse is within the pixmap area
        if not (self._pixmap_offset.x() <= pos.x() < self._pixmap_offset.x() + self.pixmap().width() and
                self._pixmap_offset.y() <= pos.y() < self._pixmap_offset.y() + self.pixmap().height()):
            self.leaveEvent(event)
            return

        # Calculate position in original image coordinates
        pixmap_pos = QPoint(
            int((pos.x() - self._pixmap_offset.x()) / self._pixmap_scale),
            int((pos.y() - self._pixmap_offset.y()) / self._pixmap_scale)
        )

        # Ensure position is within bounds
        if (0 <= pixmap_pos.x() < self.cluster_labels.shape[1] and
            0 <= pixmap_pos.y() < self.cluster_labels.shape[0]):
            cluster_idx = self.cluster_labels[pixmap_pos.y(), pixmap_pos.x()]
            group_idx = self.cluster_groups[cluster_idx]

            # Highlight all clusters in this group
            highlighted = np.full_like(self.original_img, 255)  # White background
            mask = np.isin(self.cluster_labels, [c for c, g in enumerate(self.cluster_groups) if g == group_idx])
            highlighted[mask] = self.original_img[mask]

            h, w = highlighted.shape[:2]
            bytes_per_line = 3 * w
            qimg = QImage(highlighted.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.current_pixmap = QPixmap.fromImage(qimg)
            self.setPixmap(self.scalePixmap(self.current_pixmap))

            self.clusterHovered.emit(group_idx)

    def leaveEvent(self, event):
        if self.original_img is not None:
            h, w = self.original_img.shape[:2]
            bytes_per_line = 3 * w
            qimg = QImage(self.original_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.current_pixmap = QPixmap.fromImage(qimg)
            self.setPixmap(self.scalePixmap(self.current_pixmap))
        super().leaveEvent(event)

    def resizeEvent(self, event):
        if self.current_pixmap:
            self.setPixmap(self.scalePixmap(self.current_pixmap))
        super().resizeEvent(event)