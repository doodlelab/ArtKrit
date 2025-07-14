from krita import DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Krita
from PyQt5.QtCore import Qt, QPointF, QMimeData, QEventLoop, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QLineEdit, QHBoxLayout, QSlider, QSpinBox, QSizePolicy,
    QTabWidget, QScrollArea, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPainterPath, QGuiApplication, QClipboard
from ArtKrit.script.value_color.value_color import ValueColor

import os
import sys
sys.path.append(os.path.expanduser("~/ddraw/lib/python3.10/site-packages"))
import requests

class ArtKrit(DockWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ArtKrit")
        self.preview_image = None
        self.image_file_path = None
        self.compose_lines = []
        self.value_color = ValueColor(self)

        self.setUI()


    def setUI(self):
        # Main widget and layout
        self.main_widget = QWidget()
        self.setWidget(self.main_widget)
        self.main_layout = QVBoxLayout()
        # self.main_layout.setContentsMargins(0, 0, 0, 0)
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
        # self.composition_layout.setSpacing(10)
        # self.composition_layout.setContentsMargins(0, 0, 0, 0)
        self.composition_layout.setAlignment(Qt.AlignTop)
        self.composition_tab.setLayout(self.composition_layout)
        
        # Add a button to set reference image
        self.set_reference_image_btn = QPushButton("Set Reference Image")
        self.set_reference_image_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.set_reference_image_btn.clicked.connect(self.set_reference_image)
        self.composition_layout.addWidget(self.set_reference_image_btn)

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
                    
            # Ensure the temp directory exists
            temp_dir = os.getcwd()
            if sys.platform == "darwin":  # Check if the OS is MacOS
                temp_dir = os.path.expanduser("~/Library/Application Support/krita/pykrita/artkrit")
            elif sys.platform == "sys":
                temp_dir = os.path.expanduser("~/.local/share/krita/pykrita/artkrit")

            temp_dir = os.path.join(temp_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "krita_temp_image.png")

            print(f"Sending request to server with file path: {temp_path}")
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                "http://localhost:5001/process_image", 
                json={
                    "file_path": temp_path,
                    "text_prompt": self.text_prompt_input.text(),
                    "polygon_epsilon": self.polygon_epsilon_slider.value(),
                    "custom_rectangles": custom_rectangles
                },
                headers=headers,
            )
            print(f"Server response status code: {response.status_code}")
            
            try:
                response_json = response.json()
                print(response_json)
            except:
                print("Failed to parse response as JSON")
                return
            
            
            root = document.rootNode()
            
            ## Draw polygons in krita
            ploygon_contours = response_json['ploygon_contours']
            
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
            points = response_json['points']
            
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
            self.compose_lines = response_json['composition_lines']
            self.draw_composition_lines()
        
        
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
        # self.krita_sleep(150)
            
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
        self.value_color.upload_image(half_size_path)
    
    
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
    

    def update_preview(self):
        if self.preview_image:
            pixmap = QPixmap.fromImage(self.preview_image)
            if self.preview_thirds_visible:
                pixmap = self.draw_thirds_lines(pixmap.copy())
            if self.preview_cross_visible:
                pixmap = self.draw_cross_lines(pixmap.copy())
            if self.preview_circle_visible:
                pixmap = self.draw_circle_grid(pixmap.copy())
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)

    def toggle_preview_thirds(self):
        self.preview_thirds_visible = not self.preview_thirds_visible
        self.update_preview()

    def toggle_preview_cross(self):
        self.preview_cross_visible = not self.preview_cross_visible
        self.update_preview()

    def toggle_preview_circle(self):
        self.preview_circle_visible = not self.preview_circle_visible
        self.update_preview()

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
            pen = QPen(QColor(0, 255, 0))  # Red color for lines
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
            pen = QPen(QColor(0, 255, 0))  # Blue color for circle
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
            else:
                # Toggle visibility of the existing layer
                current_visibility = third_layer.visible()
                third_layer.setVisible(not current_visibility)
                document.refreshProjection()

    def toggle_canvas_cross(self):
        document = Krita.instance().activeDocument()
        if document:
            cross_layer = document.nodeByName('Cross Grid')
            if not cross_layer:
                self.create_cross_layer()
            else:
                # Toggle visibility of the existing layer
                current_visibility = cross_layer.visible()
                cross_layer.setVisible(not current_visibility)
                document.refreshProjection()

    def toggle_canvas_circle(self):
        document = Krita.instance().activeDocument()
        if document:
            circle_layer = document.nodeByName('Circle Grid')
            if not circle_layer:
                self.create_circle_layer()
            else:
                # Toggle visibility of the existing layer
                current_visibility = circle_layer.visible()
                circle_layer.setVisible(not current_visibility)
                document.refreshProjection()

    def toggle_adaptive_grid(self):
        document = Krita.instance().activeDocument()
        if document:
            adaptive_grid_layer = document.nodeByName('Adaptive Grid')
            if adaptive_grid_layer:
                # Toggle visibility of the existing layer
                current_visibility = adaptive_grid_layer.visible()
                adaptive_grid_layer.setVisible(not current_visibility)
                document.refreshProjection()
            else:
                print("Adaptive grid layer not found")

    def toggle_contours(self):
        document = Krita.instance().activeDocument()
        if document:
            contours_layer = document.nodeByName('Contours')
            if contours_layer:
                current_visibility = contours_layer.visible()
                contours_layer.setVisible(not current_visibility)
                document.refreshProjection()
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