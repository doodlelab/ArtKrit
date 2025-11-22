from krita import DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Krita, InfoObject
from PyQt5.QtCore import Qt, QPointF, QMimeData, QEventLoop, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QHBoxLayout, QSlider, QSpinBox, QSizePolicy,
    QTabWidget, QScrollArea, QGroupBox, QDialog
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPainterPath, QGuiApplication, QClipboard
from ArtKrit.script.value_color.value_color import ValueColor
import json
from datetime import datetime
from ArtKrit.script.composition.run_models import init_models, detect, segment
from ArtKrit.script.composition.composition_utils import process_image_direct, regenerate_lines_direct

import os
import sys
sys.path.append(os.path.expanduser("~/ddraw/lib/python3.10/site-packages"))

class PreviewDialog(QDialog):
    """Popup dialog for showing the reference image with overlays"""
    def __init__(self, parent, reference_image):
        super().__init__(parent)
        self.parent_widget = parent
        self.reference_image = reference_image
        self.setWindowTitle("Reference Image Preview")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setScaledContents(False)
        layout.addWidget(self.preview_label)
        
        self.setLayout(layout)
        self.update_preview()
    
    def update_preview(self):
        """Update the preview with current overlays"""
        if self.reference_image:
            pixmap = QPixmap.fromImage(self.reference_image)
            pixmap = self.parent_widget.draw_overlays_on_pixmap(pixmap)
            
            # Scale to fit dialog while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """Handle resize events to update preview"""
        super().resizeEvent(event)
        self.update_preview()


class ArtKrit(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArtKrit")
        self.preview_image = None
        self.image_file_path = None
        self.compose_lines = []
        self.cached_points = []  # Store points for regeneration
        self.cached_polygon_contours = []  # Store polygon contours
        self.value_color = ValueColor(self)
        
        # Track overlay visibility states
        self.thirds_visible = False
        self.cross_visible = False
        self.circle_visible = False
        self.adaptive_grid_visible = False
        self.contours_visible = False
        
        print("[Plugin] Initializing models...")
        self.detector, self.segmentator, self.processor = init_models()
        print("[Plugin] Models initialized successfully")
    
        # Reference to popup dialog
        self.preview_dialog = None

        self.setUI()


    def setUI(self):
        # Main widget and layout
        self.main_widget = QWidget()
        self.setWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignTop)  # Align widgets to the top
        self.main_widget.setLayout(self.main_layout)

        # Create tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)

        # Create first tab (Composition Grid)
        self.create_composition_tab()
        self.value_color.create_value_tab() 
        self.value_color.create_color_tab()
        
        # Add tabs to tab widget
        self.tab_widget.addTab(self.composition_tab, "Composition")
        self.tab_widget.addTab(self.value_color.value_tab, "Value")
        self.tab_widget.addTab(self.value_color.color_tab, "Color")
        
        # Create a scroll area and set the main widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.main_widget)
        self.setWidget(scroll_area)
        
    
    def create_composition_tab(self):
        self.composition_tab = QWidget()
        self.composition_layout = QVBoxLayout()
        self.composition_layout.setAlignment(Qt.AlignTop)
        self.composition_tab.setLayout(self.composition_layout)
        
        # Add a button to set reference image
        self.set_reference_image_btn = QPushButton("Set Reference Image")
        self.set_reference_image_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.set_reference_image_btn.clicked.connect(self.set_reference_image)
        self.composition_layout.addWidget(self.set_reference_image_btn)

        # Add preview area for reference image
        self.preview_group = QGroupBox("Reference Preview")
        self.preview_layout = QVBoxLayout()
        self.preview_group.setLayout(self.preview_layout)
        
        # Preview label for showing the image
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setMaximumHeight(300)
        self.preview_label.setScaledContents(False)
        self.preview_label.setStyleSheet("QLabel { background-color: #2a2a2a; border: 1px solid #555; }")
        self.preview_layout.addWidget(self.preview_label)
        
        # Pop out button
        self.popout_btn = QPushButton("Pop Out Preview")
        self.popout_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.popout_btn.clicked.connect(self.toggle_preview_dialog)
        self.popout_btn.setEnabled(False)
        self.preview_layout.addWidget(self.popout_btn)
        
        self.composition_layout.addWidget(self.preview_group)

        # Create a group box for predefined grids
        self.predefined_grids_group = QGroupBox("Predefined Grid")
        self.predefined_grids_layout = QVBoxLayout()
        self.predefined_grids_group.setLayout(self.predefined_grids_layout)

        # Add Rule of Thirds button
        self.thirds_btn = QPushButton("Toggle Rule of Thirds Grid")
        self.thirds_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.thirds_btn.clicked.connect(self.toggle_canvas_thirds)
        self.predefined_grids_layout.addWidget(self.thirds_btn)

        # Add Cross Grid button
        self.cross_btn = QPushButton("Toggle Cross Grid")
        self.cross_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.cross_btn.clicked.connect(self.toggle_canvas_cross)
        self.predefined_grids_layout.addWidget(self.cross_btn)

        # Add Circle Grid button
        self.circle_btn = QPushButton("Toggle Circle Grid")
        self.circle_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.circle_btn.clicked.connect(self.toggle_canvas_circle)
        self.predefined_grids_layout.addWidget(self.circle_btn)

        # Add the group box to the composition layout
        self.composition_layout.addWidget(self.predefined_grids_group)

        # Create a group box for adaptive grid settings
        self.adaptive_grid_group = QGroupBox("Adaptive Grid")
        self.adaptive_grid_layout = QVBoxLayout()
        self.adaptive_grid_group.setLayout(self.adaptive_grid_layout)

        ## add a text input field for the user to input text prompt for GroundingDINO
        self.text_prompt_widget = QWidget()  # Create a widget to hold the label and input
        self.text_prompt_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Prevent vertical stretching
        self.text_prompt_layout = QHBoxLayout()  # Create a horizontal layout
        self.text_prompt_layout.setContentsMargins(0, 3, 0, 0)  # Small top margin for spacing
        self.text_prompt_label = QLabel("Text Prompt")  # Create a label for the text input
        self.text_prompt_input = QLineEdit()  # Create a text input field
        self.text_prompt_layout.addWidget(self.text_prompt_label)  # Add the label to the horizontal layout
        self.text_prompt_layout.addWidget(self.text_prompt_input)  # Add the input field to the horizontal layout
        self.text_prompt_widget.setLayout(self.text_prompt_layout)  # Set the layout for the widget
        self.adaptive_grid_layout.addWidget(self.text_prompt_widget)  # Add the widget to the adaptive grid layout
        
        ## add a numeric slider with value between 0 and 20, label it with "Polygon Epsilon". make them in the same row
        self.polygon_epsilon_widget = QWidget()
        self.polygon_epsilon_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Prevent vertical stretching
        self.polygon_epsilon_layout = QHBoxLayout()  # Change to QHBoxLayout to place slider and spinbox in the same row
        self.polygon_epsilon_layout.setContentsMargins(0, 3, 0, 0)  # Small top margin for spacing
        self.polygon_epsilon_label = QLabel("Polygon Epsilon")  # Initial label with default value
        self.polygon_epsilon_slider = QSlider(Qt.Horizontal)
        self.polygon_epsilon_slider.setValue(8)  # Set a default value
        self.polygon_epsilon_slider.setMinimum(0)
        self.polygon_epsilon_slider.setMaximum(20)

        self.polygon_epsilon_spinbox = QSpinBox()  # Create a spinbox
        self.polygon_epsilon_spinbox.setMinimum(0)
        self.polygon_epsilon_spinbox.setMaximum(20)
        self.polygon_epsilon_spinbox.setValue(8)  # Set a default value

        # Connect slider and spinbox to update each other
        self.polygon_epsilon_slider.valueChanged.connect(self.polygon_epsilon_spinbox.setValue)
        self.polygon_epsilon_spinbox.valueChanged.connect(self.polygon_epsilon_slider.setValue)

        self.polygon_epsilon_layout.addWidget(self.polygon_epsilon_label)
        self.polygon_epsilon_layout.addWidget(self.polygon_epsilon_slider)
        self.polygon_epsilon_layout.addWidget(self.polygon_epsilon_spinbox)
        self.polygon_epsilon_widget.setLayout(self.polygon_epsilon_layout)
        self.adaptive_grid_layout.addWidget(self.polygon_epsilon_widget)
        
        ## add a slider to change the number of lines shown in the composition grid
        self.grid_lines_widget = QWidget()
        self.grid_lines_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Prevent vertical stretching
        self.grid_lines_layout = QHBoxLayout()  # Change to QHBoxLayout to place slider and spinbox in the same row
        self.grid_lines_layout.setContentsMargins(0, 3, 0, 3)  # Small margins for spacing
        self.grid_lines_label = QLabel("Number of Grid Lines")  # Initial label without default value
        self.grid_lines_slider = QSlider(Qt.Horizontal)
        self.grid_lines_slider.setValue(2)  # Set a default value
        self.grid_lines_slider.setMinimum(1)  # Minimum 1 line
        self.grid_lines_slider.setMaximum(10)  # Maximum 10 lines

        self.grid_lines_spinbox = QSpinBox()  # Create a spinbox
        self.grid_lines_spinbox.setMinimum(1)
        self.grid_lines_spinbox.setMaximum(10)
        self.grid_lines_spinbox.setValue(2)  # Set a default value

        # Connect slider and spinbox to update each other
        self.grid_lines_slider.valueChanged.connect(self.grid_lines_spinbox.setValue)
        self.grid_lines_spinbox.valueChanged.connect(self.grid_lines_slider.setValue)

        self.grid_lines_slider.valueChanged.connect(self.draw_composition_lines)
        self.grid_lines_slider.valueChanged.connect(self.update_preview)
        self.grid_lines_layout.addWidget(self.grid_lines_label)
        self.grid_lines_layout.addWidget(self.grid_lines_slider)
        self.grid_lines_layout.addWidget(self.grid_lines_spinbox)
        self.grid_lines_widget.setLayout(self.grid_lines_layout)
        self.adaptive_grid_layout.addWidget(self.grid_lines_widget)
        
        # Create composition grid button
        self.canvas_circle_btn = QPushButton("Generate Adaptive Grid")
        self.canvas_circle_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Prevent vertical stretching
        self.canvas_circle_btn.clicked.connect(self.draw_grid)
        self.adaptive_grid_layout.addWidget(self.canvas_circle_btn)
        
        # Add NEW button to regenerate lines from current points
        self.regenerate_lines_btn = QPushButton("Regenerate Lines from Points")
        self.regenerate_lines_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.regenerate_lines_btn.clicked.connect(self.regenerate_lines_from_points)
        self.adaptive_grid_layout.addWidget(self.regenerate_lines_btn)
        
        # Add a button that to toggle the visibility of the adaptive grid
        self.toggle_adaptive_grid_btn = QPushButton("Toggle Adaptive Grid")
        self.toggle_adaptive_grid_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Prevent vertical stretching
        self.toggle_adaptive_grid_btn.clicked.connect(self.toggle_adaptive_grid)
        self.adaptive_grid_layout.addWidget(self.toggle_adaptive_grid_btn)

        # Add the adaptive grid group to the main composition layout
        self.composition_layout.addWidget(self.adaptive_grid_group)
        
        # Add a button to show the contours
        self.show_contours_btn = QPushButton("Get Composition Feedback")
        self.show_contours_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)  # Prevent vertical stretching
        self.show_contours_btn.clicked.connect(self.toggle_contours)
        self.composition_layout.addWidget(self.show_contours_btn)

    def process_image(self, image_path, text_prompt, custom_rectangles, polygon_epsilon):
        """
        Process image using direct model calls (no server needed).
        
        This replaces the old HTTP POST to /process_image endpoint.
        """
        try:
            from PIL import Image
            from .script.composition.composition_utils import load_image, DetectionResult
            import time
            
            t0 = time.time()
            
            # Load image
            if isinstance(image_path, str):
                image = load_image(image_path)
            else:
                image = image_path  # Already a PIL Image
            
            # Parse labels
            labels = [l.strip() for l in text_prompt.split(",") if l.strip()]
            threshold_bbox = 0.3
            polygon_refinement = True
            
            if (not labels) and (len(custom_rectangles) == 0):
                return {"error": "No labels or custom rectangles provided"}
            
            # Detect
            t_load = time.time()
            print(f"[Direct] Calling GroundingDINO (labels={labels}, threshold={threshold_bbox})")
            
            # Import detect and segment HERE to avoid circular imports
            from .script.composition.run_models import detect, segment
            
            detections = detect(image, labels, self.detector, threshold_bbox)
            t_detect = time.time()
            print(f"[Direct] GroundingDINO finished in {t_detect - t_load:.2f}s")
            
            # Add custom rectangles
            for custom_rectangle in custom_rectangles:
                detections.append(DetectionResult.from_dict({
                    "score": 1.0,
                    "label": "custom_rectangle",
                    "box": {
                        "xmin": int(custom_rectangle[0]),
                        "ymin": int(custom_rectangle[1]),
                        "xmax": int(custom_rectangle[2]),
                        "ymax": int(custom_rectangle[3])
                    }
                }))
            
            # Segment
            print("[Direct] Calling SAM model for segmentation")
            detections = segment(image, detections, self.segmentator, self.processor, None, polygon_refinement)
            t_segment = time.time()
            print(f"[Direct] SAM finished in {t_segment - t_detect:.2f}s")
            
            # Now call the simplified process function
            from .script.composition.composition_utils import process_image_direct
            result = process_image_direct(image, detections, polygon_epsilon)
            
            if "error" in result:
                print(f"[Plugin] Error: {result['error']}")
                return None
            
            return result
            
        except Exception as e:
            print(f"[Plugin] Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None

        
    def regenerate_lines(self, points, polygon_contours):
        """
        Regenerate composition lines from manually adjusted points.
        
        This replaces the old HTTP POST to /regenerate_lines endpoint.
        """
        try:
            lines_list = regenerate_lines_direct(points, polygon_contours)
            
            return {
                'composition_lines': lines_list,
                'num_points': len(points),
                'num_lines': len(lines_list)
            }
            
        except Exception as e:
            print(f"[Plugin] Error regenerating lines: {e}")
            import traceback
            traceback.print_exc()
            return None

    def draw_overlays_on_pixmap(self, pixmap):
        """Draw all active overlays on a pixmap"""
        if pixmap.isNull():
            return pixmap
            
        # Create a copy to draw on
        result = pixmap.copy()
        painter = QPainter(result)
        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(max(3, result.width() // 200))  # Scale pen width
        painter.setPen(pen)
        
        width = result.width()
        height = result.height()
        
        # Draw thirds grid
        if self.thirds_visible:
            for i in range(1, 3):
                x = width * i / 3
                painter.drawLine(int(x), 0, int(x), height)
            for i in range(1, 3):
                y = height * i / 3
                painter.drawLine(0, int(y), width, int(y))
        
        # Draw cross grid
        if self.cross_visible:
            x = width / 2
            painter.drawLine(int(x), 0, int(x), height)
            y = height / 2
            painter.drawLine(0, int(y), width, int(y))
        
        # Draw circle grid
        if self.circle_visible:
            center_x = width / 2
            center_y = height / 2
            radius = min(width, height) / 4
            painter.drawEllipse(int(center_x - radius), int(center_y - radius), 
                              int(radius * 2), int(radius * 2))
        
        # Draw contours
        if self.contours_visible and self.cached_polygon_contours:
            pen.setColor(QColor(0, 0, 255))
            painter.setPen(pen)
            document = Krita.instance().activeDocument()
            if document:
                scale_x = width / document.width()
                scale_y = height / document.height()
                for polygon in self.cached_polygon_contours:
                    path = QPainterPath()
                    if polygon:
                        first_point = polygon[0]
                        path.moveTo(first_point[0] * scale_x, first_point[1] * scale_y)
                        for point in polygon[1:]:
                            path.lineTo(point[0] * scale_x, point[1] * scale_y)
                        path.closeSubpath()
                        painter.drawPath(path)
        
        # Draw adaptive grid lines
        if self.adaptive_grid_visible and self.compose_lines:
            pen.setColor(QColor(0, 255, 0))
            painter.setPen(pen)
            document = Krita.instance().activeDocument()
            if document:
                scale_x = width / document.width()
                scale_y = height / document.height()
                num_lines_to_draw = min(self.grid_lines_slider.value(), len(self.compose_lines))
                for line in self.compose_lines[:num_lines_to_draw]:
                    p1, p2 = line
                    painter.drawLine(int(p1[0] * scale_x), int(p1[1] * scale_y),
                                   int(p2[0] * scale_x), int(p2[1] * scale_y))
        
        painter.end()
        return result


    def update_preview(self):
        """Update the preview in both the dock and popup dialog"""
        if self.preview_image:
            pixmap = QPixmap.fromImage(self.preview_image)
            pixmap = self.draw_overlays_on_pixmap(pixmap)
            
            # Update dock preview
            scaled_pixmap = pixmap.scaled(
                self.preview_label.width(),
                self.preview_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
            
            # Update popup dialog if it exists
            if self.preview_dialog and self.preview_dialog.isVisible():
                self.preview_dialog.update_preview()


    def toggle_preview_dialog(self):
        """Toggle the preview popup dialog"""
        if self.preview_dialog is None or not self.preview_dialog.isVisible():
            self.preview_dialog = PreviewDialog(self, self.preview_image)
            self.preview_dialog.show()
            self.popout_btn.setText("Close Pop Out")
        else:
            self.preview_dialog.close()
            self.preview_dialog = None
            self.popout_btn.setText("Pop Out Preview")


    def read_points_from_layer(self):
        """Read point positions from the Points vector layer"""
        document = Krita.instance().activeDocument()
        if not document:
            return []
        
        points_layer = document.nodeByName('Points')
        if not points_layer or points_layer.type() != "vectorlayer":
            print("Points layer not found or not a vector layer")
            return []
        
        points = []
        points_per_inch = 72.0
        
        # Extract circle centers from SVG shapes
        for shape in points_layer.shapes():
            if shape.type() == "KoPathShape":
                # Get the bounding box of the circle
                bbox = shape.boundingBox()
                # Calculate center point
                center_x = (bbox.topLeft().x() + bbox.bottomRight().x()) / 2
                center_y = (bbox.topLeft().y() + bbox.bottomRight().y()) / 2
                # Convert from points to pixels
                x = int(center_x * document.xRes() / points_per_inch)
                y = int(center_y * document.yRes() / points_per_inch)
                points.append([x, y])
        
        return points


    def regenerate_lines_from_points(self):
        """Regenerate composition lines based on current point positions without calling models"""
        document = Krita.instance().activeDocument()
        if not document:
            print("No active document")
            return
        
        # Read current point positions from the Points layer
        current_points = self.read_points_from_layer()
        
        if not current_points:
            print("No points found in Points layer")
            return
        
        if not self.cached_polygon_contours:
            print("No cached polygon contours. Please generate adaptive grid first.")
            return
        
        print(f"Found {len(current_points)} points, regenerating lines...")
        
        # Call the direct function (no server needed)
        result = self.regenerate_lines(current_points, self.cached_polygon_contours)
        
        if not result:
            print("Failed to regenerate lines")
            return
        
        self.compose_lines = result['composition_lines']
        print(f"Generated {len(self.compose_lines)} composition lines")
        
        # Redraw the composition lines
        self.draw_composition_lines()
        self.update_preview()
        self.value_color.append_log_entry("regenerate lines", 
            f"Regenerated {len(self.compose_lines)} composition lines from {len(current_points)} points")

    def draw_grid(self):
        document = Krita.instance().activeDocument()
        if document:
            w = Krita.instance().activeWindow()
            v = w.activeView()
            selected_nodes = v.selectedNodes()
            
            ## get all the rectangles from the selected nodes
            custom_rectangles = []
            for node in selected_nodes:
                print(f"Selected node: {node.name()} of type {node.type()}")
                # Get all rectangles in the vector layer
                if node.type() == "vectorlayer":
                    for shape in node.shapes():
                        if shape.type() == "KoPathShape":
                            # Get the bounding box in points
                            bbox = shape.boundingBox()
                            # Convert from points to pixels (72 points per inch)
                            points_per_inch = 72.0
                            x1 = int(bbox.topLeft().x() * document.xRes() / points_per_inch)
                            y1 = int(bbox.topLeft().y() * document.yRes() / points_per_inch)
                            x2 = int(bbox.bottomRight().x() * document.xRes() / points_per_inch)
                            y2 = int(bbox.bottomRight().y() * document.yRes() / points_per_inch)
                            custom_rectangles.append((x1, y1, x2, y2))
                            
            ## process the paint layer
            reference_layer = document.nodeByName("Reference Image")
            if not reference_layer:
                print("No reference image found")
                return
            
            width = document.width()
            height = document.height()
                    
            # Ensure the temp directory exists and save the reference image
            temp_dir = os.getcwd()
            if sys.platform == "darwin":  # Check if the OS is MacOS
                temp_dir = os.path.expanduser("~/Library/Application Support/krita/pykrita/artkrit")
            elif sys.platform == "sys":
                temp_dir = os.path.expanduser("~/.local/share/krita/pykrita/artkrit")

            temp_dir = os.path.join(temp_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "krita_temp_image.png")

            print(f"Processing image with file path: {temp_path}")
            
            # Call the direct processing function (no server needed)
            result = self.process_image(
                image_path=temp_path,
                text_prompt=self.text_prompt_input.text(),
                custom_rectangles=custom_rectangles,
                polygon_epsilon=self.polygon_epsilon_slider.value()
            )
            
            if not result:
                print("Failed to process image")
                return
            
            print(f"Processing complete: {len(result.get('ploygon_contours', []))} polygons, {len(result.get('points', []))} points, {len(result.get('composition_lines', []))} lines")
            
            root = document.rootNode()
            
            ## Draw polygons in krita
            ploygon_contours = result['ploygon_contours']
            self.cached_polygon_contours = ploygon_contours  # Cache for regeneration
            
            # Create contours vector layer
            contour_layer = document.nodeByName('Contours')
            if contour_layer is None:
                contour_layer = document.createVectorLayer('Contours')
                root.addChildNode(contour_layer, None)
            
            # Remove existing shapes
            for shape in contour_layer.shapes():
                shape.remove()
            
            document.setActiveNode(contour_layer)
            
            # Create SVG content for all polygons
            svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">'''
            
            for polygon in ploygon_contours:
                # Create path data for polygon
                points = [f"{point[0]},{point[1]}" for point in polygon]
                path_data = "M " + " L ".join(points) + " Z"  # Z closes the path
                svg_content += f'<path d="{path_data}" stroke="#0000FF" stroke-width="10" fill="none"/>'
            
            svg_content += '</svg>'
            
            # Add SVG shapes to the layer
            contour_layer.addShapesFromSvg(svg_content)
            contour_layer.setVisible(False)
            
            ## Draw points in krita
            points = result['points']
            self.cached_points = points  # Cache for reference
            
            # Create points vector layer
            points_layer = document.nodeByName('Points')
            if points_layer is None:
                points_layer = document.createVectorLayer('Points')
                root.addChildNode(points_layer, None)
            
            # Remove existing shapes
            for shape in points_layer.shapes():
                shape.remove()
            
            document.setActiveNode(points_layer)
            
            # Create SVG content for all points
            svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">'''
            
            for point in points:
                x, y = point
                # Draw a small circle for each point
                svg_content += f'<circle cx="{x}" cy="{y}" r="10" fill="#FF0000" stroke="none"/>'
            
            svg_content += '</svg>'
            
            # Add SVG shapes to the layer
            points_layer.addShapesFromSvg(svg_content)
            points_layer.setVisible(True)
            
            ## Draw lines in krita
            self.compose_lines = result['composition_lines']
            self.draw_composition_lines()
            self.update_preview()
            self.value_color.append_log_entry("generate adaptive grid", 
                f'Generating adaptive grid with prompt: {self.text_prompt_input.text()} and polygon epsilon: {self.polygon_epsilon_slider.value()} and number of lines: {self.grid_lines_slider.value()}')

        
    def draw_composition_lines(self):        
        document = Krita.instance().activeDocument()
        if not document or len(self.compose_lines) == 0:
            return
        
        width = document.width()
        height = document.height()
        root = document.rootNode()
                    
        # Create composition vector layer
        compose_layer = document.nodeByName('Adaptive Grid')
        if compose_layer is None:
            compose_layer = document.createVectorLayer('Adaptive Grid')
            root.addChildNode(compose_layer, None)
        
        for shape in compose_layer.shapes():
            shape.remove()
        
        document.setActiveNode(compose_layer)
            
        # Create SVG content for all lines
        svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}px" height="{height}px" viewBox="0 0 {width} {height}">'''
        num_lines_to_draw = self.grid_lines_slider.value()
        num_lines_to_draw = min(num_lines_to_draw, len(self.compose_lines))
        for (i, line) in enumerate(self.compose_lines[:num_lines_to_draw]):
            p1, p2 = line
            svg_content += f'<path d="M {p1[0]} {p1[1]} L {p2[0]} {p2[1]}" stroke="#00FF00" stroke-width="15" fill="none"/>'
        svg_content += '</svg>'
        
        compose_layer.addShapesFromSvg(svg_content)
        compose_layer.setVisible(True)

        # Refresh the document
        document.refreshProjection()
        
    
    def set_reference_image(self):
        # First check if there is a reference image layer already
        document = Krita.instance().activeDocument()
        if not document:
            return
        
        reference_layer = document.nodeByName('Reference Image')
        if not reference_layer:
            # Open file dialog to select image
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(None, "Select Reference Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
            
            if not file_path:
                return
                
            # Get active document
            document = Krita.instance().activeDocument()
            if not document:
                return
                
            # Read the image
            image = QImage(file_path)
            if image.isNull():
                return
                        
            # Get document dimensions
            doc_width = document.width()
            doc_height = document.height()
            
            # Calculate scaling to fit image within document bounds while preserving aspect ratio
            image_aspect = image.width() / image.height()
            doc_aspect = doc_width / doc_height
            
            if image_aspect > doc_aspect:
                # Image is wider relative to height - scale to fit width
                scaled_width = doc_width
                scaled_height = int(doc_width / image_aspect)
            else:
                # Image is taller relative to width - scale to fit height
                scaled_height = doc_height
                scaled_width = int(doc_height * image_aspect)
                
            # Scale the image
            scaled_image = image.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # Create paint layer
            root = document.rootNode()
            reference_layer = document.createNode("Reference Image", "paintlayer")
            root.addChildNode(reference_layer, None)
            
            # Calculate position to center the image
            x = int((doc_width - scaled_width) / 2)
            y = int((doc_height - scaled_height) / 2)
            
            # Create a temporary QImage with the correct size and format
            temp_image = QImage(doc_width, doc_height, QImage.Format_ARGB32)
            temp_image.fill(Qt.transparent)
            
            # Draw the scaled image onto the temporary image
            painter = QPainter(temp_image)
            painter.drawImage(x, y, scaled_image)
            painter.end()
            
            # Convert to bytes and set as pixel data
            ptr = temp_image.bits()
            ptr.setsize(temp_image.byteCount())
            byte_array = bytes(ptr)
            reference_layer.setPixelData(byte_array, 0, 0, doc_width, doc_height)
            
            # Refresh document
            document.refreshProjection()
        
        temp_path, half_size_path = self.write_layer_to_temp(reference_layer)
        
        # Load the image for preview
        self.preview_image = QImage(temp_path)
        if not self.preview_image.isNull():
            self.update_preview()
            self.popout_btn.setEnabled(True)
        
        self.value_color.upload_image(half_size_path)
        self.value_color.append_log_entry("set ref img composition", "Setting reference image for composition")
    
    
    def write_layer_to_temp(self, layer):
        document = Krita.instance().activeDocument()
        if not document:
            return
        
        # Get active node
        node = document.nodeByName(layer.name())
        if not node:
            return
        
        # Read the layer
        width = document.width()
        height = document.height()
                    
        # Get the layer dimensions
        pixel_data = node.pixelData(0, 0, width, height)

        # Create a QImage and copy the pixel data into it
        temp_image = QImage(pixel_data, width, height, QImage.Format_RGBA8888).rgbSwapped()
        
        # Save the reference image to temp directory
        temp_dir = os.getcwd()
        if sys.platform == "darwin":  # Check if the OS is MacOS
            temp_dir = os.path.expanduser("~/Library/Application Support/krita/pykrita/artkrit")
        elif sys.platform == "sys":
            temp_dir = os.path.expanduser("~/.local/share/krita/pykrita/artkrit")

        temp_dir = os.path.join(temp_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Save the reference image
        temp_path = os.path.join(temp_dir, "krita_temp_image.png")
        if temp_image.save(temp_path):
            print(f"Image saved successfully to {temp_path}")
        else:
            print("Failed to save image")
            
        # also write an image half the size
        half_size_path = os.path.join(temp_dir, "krita_temp_image_half_size.png")
        half_size_image = temp_image.scaled(temp_image.width() // 2, temp_image.height() // 2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if half_size_image.save(half_size_path):
            print(f"Half size image saved successfully to {half_size_path}")
        else:
            print("Failed to save half size image") 

        return temp_path, half_size_path


    def create_thirds_layer(self):
        document = Krita.instance().activeDocument()
        if document:
            # Create a new transparent layer for the lines
            root = document.rootNode()
            self.thirds_layer = document.createNode("Rule of Thirds Grid", "paintlayer")
            root.addChildNode(self.thirds_layer, None)

            # Get document dimensions
            width = document.width()
            height = document.height()

            # Create a transparent RGBA image for the layer
            image = QImage(width, height, QImage.Format_RGBA8888)
            image.fill(Qt.transparent)

            # Draw the lines
            painter = QPainter(image)
            pen = QPen(QColor(0, 255, 0))  # Green color for lines
            pen.setWidth(15)
            painter.setPen(pen)

            # Draw vertical lines
            for i in range(1, 3):
                x = width * i / 3
                painter.drawLine(int(x), 0, int(x), height)

            # Draw horizontal lines
            for i in range(1, 3):
                y = height * i / 3
                painter.drawLine(0, int(y), width, int(y))

            painter.end()

            # Convert QImage to bytes and set as pixel data
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            byte_array = bytes(ptr)

            # Set the pixel data on the layer
            self.thirds_layer.setPixelData(byte_array, 0, 0, width, height)

            # Make sure the layer is visible initially
            self.thirds_layer.setVisible(True)

            # Refresh the document
            document.refreshProjection()

    def create_cross_layer(self):
        document = Krita.instance().activeDocument()
        if document:
            # Create a new transparent layer for the lines
            root = document.rootNode()
            self.cross_layer = document.createNode("Cross Grid", "paintlayer")
            root.addChildNode(self.cross_layer, None)

            # Get document dimensions
            width = document.width()
            height = document.height()

            # Create a transparent RGBA image for the layer
            image = QImage(width, height, QImage.Format_RGBA8888)
            image.fill(Qt.transparent)

            # Draw the lines
            painter = QPainter(image)
            pen = QPen(QColor(0, 255, 0))  # Green color for lines
            pen.setWidth(15)
            painter.setPen(pen)

            # Draw vertical line (middle)
            x = width / 2
            painter.drawLine(int(x), 0, int(x), height)

            # Draw horizontal line (middle)
            y = height / 2
            painter.drawLine(0, int(y), width, int(y))

            painter.end()

            # Convert QImage to bytes and set as pixel data
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            byte_array = bytes(ptr)

            # Set the pixel data on the layer
            self.cross_layer.setPixelData(byte_array, 0, 0, width, height)

            # Make sure the layer is visible initially
            self.cross_layer.setVisible(True)

            # Refresh the document
            document.refreshProjection()

    def create_circle_layer(self):
        document = Krita.instance().activeDocument()
        if document:
            # Create a new transparent layer for the circle
            root = document.rootNode()
            self.circle_layer = document.createNode("Circle Grid", "paintlayer")
            root.addChildNode(self.circle_layer, None)

            # Get document dimensions
            width = document.width()
            height = document.height()

            # Create a transparent RGBA image for the layer
            image = QImage(width, height, QImage.Format_RGBA8888)
            image.fill(Qt.transparent)

            # Draw the circle
            painter = QPainter(image)
            pen = QPen(QColor(0, 255, 0))  # Green color for circle
            pen.setWidth(15)
            painter.setPen(pen)

            # Draw a circle in the center
            center_x = width / 2
            center_y = height / 2
            radius = min(width, height) / 4  # Adjust radius as needed
            painter.drawEllipse(int(center_x - radius), int(center_y - radius), int(radius * 2), int(radius * 2))

            painter.end()

            # Convert QImage to bytes and set as pixel data
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            byte_array = bytes(ptr)

            # Set the pixel data on the layer
            self.circle_layer.setPixelData(byte_array, 0, 0, width, height)

            # Make sure the layer is visible initially
            self.circle_layer.setVisible(True)

            # Refresh the document
            document.refreshProjection()

    def toggle_canvas_thirds(self):
        document = Krita.instance().activeDocument()
        if document:
            third_layer = document.nodeByName('Rule of Thirds Grid')
            if not third_layer:
                self.create_thirds_layer()
                self.thirds_visible = True
                self.value_color.append_log_entry("toggle rule of thirds grid", "Toggling rule of thirds grid")
            else:
                # Toggle visibility of the existing layer
                current_visibility = third_layer.visible()
                third_layer.setVisible(not current_visibility)
                self.thirds_visible = not current_visibility
                document.refreshProjection()
                self.value_color.append_log_entry("toggle rule of thirds grid", "Toggling rule of thirds grid")
            self.update_preview()

    def toggle_canvas_cross(self):
        document = Krita.instance().activeDocument()
        if document:
            cross_layer = document.nodeByName('Cross Grid')
            if not cross_layer:
                self.create_cross_layer()
                self.cross_visible = True
                self.value_color.append_log_entry("toggle cross grid", "Toggling cross grid")
            else:
                # Toggle visibility of the existing layer
                current_visibility = cross_layer.visible()
                cross_layer.setVisible(not current_visibility)
                self.cross_visible = not current_visibility
                document.refreshProjection()
                self.value_color.append_log_entry("toggle cross grid","Toggling cross grid")
            self.update_preview()

    def toggle_canvas_circle(self):
        document = Krita.instance().activeDocument()
        if document:
            circle_layer = document.nodeByName('Circle Grid')
            if not circle_layer:
                self.create_circle_layer()
                self.circle_visible = True
                self.value_color.append_log_entry("toggle circle grid", "Toggling circle grid")
            else:
                # Toggle visibility of the existing layer
                current_visibility = circle_layer.visible()
                circle_layer.setVisible(not current_visibility)
                self.circle_visible = not current_visibility
                document.refreshProjection()
                self.value_color.append_log_entry("toggle circle grid","Toggling circle grid")
            self.update_preview()

    def toggle_adaptive_grid(self):
        document = Krita.instance().activeDocument()
        if document:
            adaptive_grid_layer = document.nodeByName('Adaptive Grid')
            if adaptive_grid_layer:
                # Toggle visibility of the existing layer
                current_visibility = adaptive_grid_layer.visible()
                adaptive_grid_layer.setVisible(not current_visibility)
                self.adaptive_grid_visible = not current_visibility
                document.refreshProjection()
                self.value_color.append_log_entry("toggle adaptive grid", "Toggling adaptive grid")
                self.update_preview()
            else:
                print("Adaptive grid layer not found")

    def toggle_contours(self):
        document = Krita.instance().activeDocument()
        if document:
            contours_layer = document.nodeByName('Contours')
            if contours_layer:
                current_visibility = contours_layer.visible()
                contours_layer.setVisible(not current_visibility)
                self.contours_visible = not current_visibility
                document.refreshProjection()
                self.value_color.append_log_entry("contours feedback", "Toggling contours visibility")
                self.update_preview()
            else:
                print("Contours layer not found")

    def canvasChanged(self, canvas):
        # Reset layer references when canvas changes
        self.thirds_layer = None
        self.cross_layer = None
        self.circle_layer = None
        
    def krita_sleep(self, value):
        loop = QEventLoop()
        QTimer.singleShot(value, loop.quit)
        loop.exec()


# Register the docker with Krita
Krita.instance().addDockWidgetFactory(
    DockWidgetFactory("ArtKrit", DockWidgetFactoryBase.DockRight, ArtKrit)
)