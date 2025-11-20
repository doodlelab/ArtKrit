import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtWidgets import (QLabel, QScrollArea, QVBoxLayout, QWidget, 
                             QHBoxLayout, QPushButton, QColorDialog, QSlider)
from PyQt5.QtGui import QImage, QPixmap, QColor

class ClusterHoverLabel(QLabel):
    clusterHovered = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.original_img = None
        self.cluster_labels = None
        self.dominant_colors = None
        self.cluster_groups = None
        self.current_pixmap = None
        self._pixmap_offset = QPoint(0, 0)  
        self._pixmap_scale = 1.0
        self.background_color = (255, 255, 255)  # Default white

    def setBackgroundColor(self, color):
        """Set the background color for highlighting (RGB tuple)"""
        self.background_color = color

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

        # Get available size safely
        try:
            parent_widget = self.parent()
            if parent_widget and hasattr(parent_widget, 'width') and hasattr(parent_widget, 'height'):
                available_width = max(100, parent_widget.width() - 20)
                available_height = parent_widget.height() - 20
            else:
                # Fallback values if parent is not available
                available_width = 400
                available_height = 300
        except RuntimeError:
            # Parent widget has been deleted, use fallback values
            available_width = 400
            available_height = 300

        # Calculate scaled size maintaining aspect ratio
        pixmap_ratio = pixmap.width() / pixmap.height()
        available_ratio = available_width / available_height

        if pixmap_ratio > available_ratio:
            # Width is the limiting factor
            scaled_width = available_width
            scaled_height = int(scaled_width / pixmap_ratio)
        else:
            # Height is the limiting factor
            scaled_height = available_height
            scaled_width = int(scaled_height * pixmap_ratio)

        # Store scaling factor and offset for accurate coordinate mapping
        self._pixmap_scale = scaled_width / pixmap.width()

        # Calculate the centered position
        self._pixmap_offset = QPoint(
            (self.width() - scaled_width) // 2 if hasattr(self, 'width') else 0,
            (self.height() - scaled_height) // 2 if hasattr(self, 'height') else 0
        )

        return pixmap.scaled(
            scaled_width, scaled_height,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

    def mouseMoveEvent(self, event):
        try:
            if self.original_img is None or self.current_pixmap is None:
                return

            # Convert mouse position to pixmap coordinates
            pos = event.pos()

            # Get current pixmap dimensions
            if not self.pixmap() or self.pixmap().isNull():
                return

            pixmap_width = self.pixmap().width()
            pixmap_height = self.pixmap().height()

            # Check if mouse is within the pixmap area
            if not (self._pixmap_offset.x() <= pos.x() < self._pixmap_offset.x() + pixmap_width and
                    self._pixmap_offset.y() <= pos.y() < self._pixmap_offset.y() + pixmap_height):
                self.leaveEvent(event)
                return

            # Calculate position in original image coordinates
            pixmap_x = pos.x() - self._pixmap_offset.x()
            pixmap_y = pos.y() - self._pixmap_offset.y()
            
            orig_x = int(pixmap_x / self._pixmap_scale)
            orig_y = int(pixmap_y / self._pixmap_scale)

            # Ensure position is within bounds
            if (0 <= orig_x < self.cluster_labels.shape[1] and
                0 <= orig_y < self.cluster_labels.shape[0]):
                cluster_idx = self.cluster_labels[orig_y, orig_x]
                group_idx = self.cluster_groups[cluster_idx]

                # Highlight all clusters in this group with custom background color
                highlighted = np.full_like(self.original_img, self.background_color, dtype=np.uint8)
                mask = np.isin(self.cluster_labels, [c for c, g in enumerate(self.cluster_groups) if g == group_idx])
                highlighted[mask] = self.original_img[mask]

                h, w = highlighted.shape[:2]
                bytes_per_line = 3 * w
                qimg = QImage(highlighted.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.current_pixmap = QPixmap.fromImage(qimg)
                self.setPixmap(self.scalePixmap(self.current_pixmap))

                self.clusterHovered.emit(group_idx)
        except RuntimeError:
            # Widget has been deleted, ignore the event
            pass

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


class ColorSeparationTool:
    def __init__(self, parent):
        self.parent = parent
        self.current_image = None
        self.current_labels = None
        self.current_colors = None
        self.current_groups = None
        self.image_label = None
        self.background_color = QColor(255, 255, 255)  # Default white
        self.bg_color_button = None
        self.color_groups_slider = None
        self.color_groups_label = None
        self.num_color_groups = 15  # Default number of color groups
        
    def cleanup(self):
        """Clean up resources to prevent memory leaks"""
        if self.image_label:
            try:
                self.image_label.deleteLater()
            except:
                pass
            self.image_label = None
        
        if self.bg_color_button:
            try:
                self.bg_color_button.deleteLater()
            except:
                pass
            self.bg_color_button = None
        
        if self.color_groups_slider:
            try:
                self.color_groups_slider.deleteLater()
            except:
                pass
            self.color_groups_slider = None
        
        if self.color_groups_label:
            try:
                self.color_groups_label.deleteLater()
            except:
                pass
            self.color_groups_label = None
        
        self.current_image = None
        self.current_labels = None
        self.current_colors = None
        self.current_groups = None
        
    def create_color_separation_ui(self):
        """Create the UI components for color separation"""
        # Main container
        main_container = QWidget(self.parent.color_tab)
        main_layout = QVBoxLayout(main_container)
        
        # Color picker controls
        controls_layout = QHBoxLayout()
        
        # Background color button
        self.bg_color_button = QPushButton("Background Color")
        self.bg_color_button.clicked.connect(self.choose_background_color)
        self.update_color_button_style()
        controls_layout.addWidget(self.bg_color_button)
        
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # Color groups slider
        slider_layout = QHBoxLayout()
        self.color_groups_label = QLabel(f"Color Groups: {self.num_color_groups}")
        slider_layout.addWidget(self.color_groups_label)
        
        self.color_groups_slider = QSlider(Qt.Horizontal)
        self.color_groups_slider.setMinimum(3)
        self.color_groups_slider.setMaximum(30)
        self.color_groups_slider.setValue(self.num_color_groups)
        self.color_groups_slider.setTickPosition(QSlider.TicksBelow)
        self.color_groups_slider.setTickInterval(3)
        self.color_groups_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.color_groups_slider)
        
        main_layout.addLayout(slider_layout)
        
        # Scrollable image container
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        img_cont = QWidget()
        v = QVBoxLayout(img_cont)
        self.image_label = ClusterHoverLabel(img_cont)
        self.image_label.setMinimumSize(200, 200)
        self.image_label.clusterHovered.connect(self.update_cluster_info)
        v.addWidget(self.image_label)

        scroll_area.setWidget(img_cont)
        main_layout.addWidget(scroll_area)
        
        return main_container, self.image_label
    
    def on_slider_changed(self, value):
        """Handle slider value change"""
        self.num_color_groups = value
        if self.color_groups_label:
            self.color_groups_label.setText(f"Color Groups: {value}")
        
        # Reprocess the image with new number of color groups
        self.update_cluster_count()
    
    def choose_background_color(self):
        """Open color picker dialog to choose background color"""
        color = QColorDialog.getColor(self.background_color, self.parent, "Choose Background Color")
        
        if color.isValid():
            self.background_color = color
            self.update_color_button_style()
            
            # Update the image label's background color
            if self.image_label:
                self.image_label.setBackgroundColor((color.red(), color.green(), color.blue()))
    
    def update_color_button_style(self):
        """Update the button style to show the current background color"""
        if self.bg_color_button:
            r, g, b = self.background_color.red(), self.background_color.green(), self.background_color.blue()
            # Calculate contrast color for text (black or white)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            text_color = "black" if brightness > 128 else "white"
            
            self.bg_color_button.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); "
                f"color: {text_color}; "
                f"border: 1px solid #888; "
                f"padding: 5px 10px; "
                f"border-radius: 3px;"
            )
        
    def process_reference_image(self, color_reference_image):
        """Process the stored reference image for color analysis"""
        if color_reference_image is not None:
            # Convert color space appropriately
            if len(color_reference_image.shape) == 2:  # Grayscale
                self.current_image = cv2.cvtColor(color_reference_image, cv2.COLOR_GRAY2RGB)
            elif color_reference_image.shape[2] == 4:  # RGBA
                self.current_image = cv2.cvtColor(color_reference_image, cv2.COLOR_BGRA2RGB)
            else:  # BGR
                self.current_image = cv2.cvtColor(color_reference_image, cv2.COLOR_BGR2RGB)
            
            self.update_cluster_count()
             
    def update_cluster_count(self):
        """Recompute dominant color clusters and update the image label accordingly."""
        if self.current_image is None:
            return

        # Check if image_label exists before using it
        if self.image_label is None:
            return

        self.parent.color_data.reference_dominant = self.parent.color_data.extract_dominant(
            self.current_image,
            num_values=self.num_color_groups
        )
        dominant_colors = self.parent.color_data.reference_dominant

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

        # Send to display - with error handling
        try:
            self.image_label.setImageData(
                self.current_image,
                self.current_labels,
                self.current_colors,
                self.current_groups
            )
        except RuntimeError:
            # Image label has been deleted, recreate it
            pass

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