from krita import Krita, ManagedColor, InfoObject
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QPushButton
from PyQt5.QtGui import QColor
import numpy as np
import os
import sys
from PIL import Image
import json
from datetime import datetime


class LassoFillTool:
    def __init__(self, parent):
        self.parent = parent
        self.currentFillColor = None
        self.selectionTimer = QTimer()
        self.selectionTimer.setSingleShot(True)
        self.selectionTimer.timeout.connect(self.checkSelection)
        self.isActive = False

    def get_json_path(self):
        """Get the path to the logs JSON file"""
        home_dir = os.path.expanduser("~")
        logs_folder = os.path.join(home_dir, "ArtKrit_logs")  
        os.makedirs(logs_folder, exist_ok=True)
        return os.path.join(logs_folder, "logs.json")
    
    def append_log_entry(self, action, message):
        """Append a log entry to the JSON log file"""
        self.save_png_on_button_press(action)
        json_path = self.get_json_path()

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"logs": []}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data["logs"].append({
            "timestamp": timestamp,
            "action": action,
            "message": message
        })

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"Logged: {action}")

    def save_png_on_button_press(self, action):
        """Save the current document as PNG when a button is pressed"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_action = action.replace(" ", "_")

        home_dir = os.path.expanduser("~")
        base_folder = os.path.join(home_dir, "ArtKrit_logs")
        images_folder = os.path.join(base_folder, "canvas_images")
        os.makedirs(images_folder, exist_ok=True)

        doc = Krita.instance().activeDocument()
        if doc is not None:
            current_path = doc.fileName()
            if not current_path:
                print("Please save your document first.")
                return
            
            png_path = os.path.join(images_folder, f"{timestamp}_{safe_action}.png")
            
            doc.setBatchmode(True)
            options = InfoObject()
            options.setProperty('compression', 5)
            options.setProperty('alpha', True)
            doc.exportImage(png_path, options)
            doc.setBatchmode(False)
        
    # In lasso_fill_tool.py
    def create_fill_widgets(self):
        """Create the fill options UI components"""
        fill_grp = QGroupBox("Fill Options")
        fv = QVBoxLayout(fill_grp)
        
        # Create the fill color button
        self.fillColorButton = QPushButton("Select Fill Color")
        self.fillColorButton.clicked.connect(self.selectFillColor)  # Connect to lasso tool method
        self.fillColorButton.setStyleSheet("background-color: #ffffff;")
        
        # Create the fill button
        self.fillButton = QPushButton("Fill Selection")
        self.fillButton.clicked.connect(self.fillSelection)  # Connect to lasso tool method
        self.fillButton.setEnabled(False)
        
        # Create the deactivate button
        self.deactivateButton = QPushButton("Deactivate Lasso Tool")
        self.deactivateButton.clicked.connect(self.deactivateLassoTool)
        
        fv.addWidget(self.fillColorButton)
        fv.addWidget(self.fillButton)
        fv.addWidget(self.deactivateButton)
        fill_grp.setLayout(fv)
        fill_grp.setVisible(False)
        
        return fill_grp, self.fillColorButton, self.fillButton

    def activateLassoTool(self):
        """Activate Krita's lasso selection tool and prepare the fill options UI."""
        krita_instance = Krita.instance()
        
        action = krita_instance.action('KisToolSelectContiguous')
        if action:
            action.trigger()
            self.isActive = True
            self.parent.lassoButton.setStyleSheet("background-color: #AED6F1;")
            
            self.parent.fillGroup.setVisible(True)
            
            self.selectionTimer.start(500)
            
            QTimer.singleShot(500, lambda: self.parent.lassoButton.setStyleSheet(""))
            self.append_log_entry("lasso tool", "Lasso tool activated")
    
    def deactivateLassoTool(self):
        """Deactivate the lasso tool, clear selection, and hide the fill group."""
        krita_instance = Krita.instance()
        doc = krita_instance.activeDocument()
        
        # Clear any active selection
        if doc and doc.selection():
            doc.setSelection(None)
            doc.refreshProjection()
        
        # Stop the selection timer
        self.selectionTimer.stop()
        
        # Hide the fill group
        self.parent.fillGroup.setVisible(False)
        
        # Reset state
        self.isActive = False
        self.fillButton.setEnabled(False)
        
        # Reset button style
        self.parent.lassoButton.setStyleSheet("")
        
        # Switch back to a default tool (e.g., freehand brush)
        defaultToolAction = krita_instance.action('KritaShape/KisToolBrush')
        if defaultToolAction:
            defaultToolAction.trigger()
        
        print("Lasso tool deactivated")
    
    def selectFillColor(self):
        """Open the HS picker seeded by the current selection's average value."""
        # Extract the dominant value from the selection
        doc = Krita.instance().activeDocument()
        if doc:
            selection = doc.selection()
            if selection:
                node = doc.activeNode()
                average_value = self.extractAverageValueFromSelection(node, selection)
                
                if average_value is not None:
                    # Open the color picker with the extracted value
                    from ..value_color import CustomHSColorPickerDialog
                    dialog = CustomHSColorPickerDialog(self.parent, average_value)
                    dialog.exec_()
                    
                    selectedColor = dialog.selectedColor()
                    if selectedColor.isValid():
                        self.currentFillColor = selectedColor
                        self.fillColorButton.setStyleSheet(f"background-color: {selectedColor.name()};")
                else:
                    # Show error message or use default value
                    print("Warning: Could not extract value from selection. Using default value.")
                    self.parent.color_feedback_label.setText("⚠️ Could not extract value from selection. Please ensure you have a valid selection with visible pixels.")
                    
                    # Use a reasonable default value (e.g., mid-tone)
                    from ..value_color import CustomHSColorPickerDialog
                    dialog = CustomHSColorPickerDialog(self.parent, 128)
                    dialog.exec_()
                    
                    selectedColor = dialog.selectedColor()
                    if selectedColor.isValid():
                        self.currentFillColor = selectedColor
                        self.fillColorButton.setStyleSheet(f"background-color: {selectedColor.name()};")
                        self.append_log_entry("lasso fill color", f"Selected lasso fill color: {selectedColor.name()}")

        
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
                self.append_log_entry("lasso fill initiated", f"Initiated lasso fill with color: {new_color.name()}")
                
            else:
                print("Could not find fill tool action")

                

    def triggerFillForeground(self, krita_instance):
        """Triggers the fill_foreground action after the fill tool is activated."""
        fillAction = krita_instance.action('fill_foreground')
        if fillAction:
            fillAction.trigger()
            

    def extractAverageValueFromSelection(self, node, selection):
        """
        Extracts the dominant brightness (value) from the selected area using the HSV color space.
        Returns None if no valid value can be extracted.
        """
        try:
            print("Extracting pixel data from selection...")
            
            # Check if selection is valid
            if not selection or selection.width() == 0 or selection.height() == 0:
                print("Invalid or empty selection")
                return None
                
            # Get the pixel data from the selected area
            pixel_data = node.projectionPixelData(
                selection.x(), selection.y(), selection.width(), selection.height()
            ).data()
            
            if not pixel_data or len(pixel_data) == 0:
                print("No pixel data available")
                return None
                
            pixels = []
            for i in range(0, len(pixel_data), 4):
                # Skip if we don't have enough data
                if i + 3 >= len(pixel_data):
                    break
                    
                r = pixel_data[i]
                g = pixel_data[i + 1]
                b = pixel_data[i + 2]
                # Only add pixels that aren't completely transparent (alpha > 0)
                a = pixel_data[i + 3]
                if a > 0:  # Only consider non-transparent pixels
                    pixels.append((r, g, b))
            
            if not pixels:
                print("No non-transparent pixels in selection")
                return None
                
            # Calculate the frequency of each brightness (value) level
            value_counts = {}
            for r, g, b in pixels:
                # Convert RGB to HSV
                color = QColor(r, g, b)
                if color.isValid():
                    h, s, v, _ = color.getHsv()
                    
                    # Count the frequency of each value
                    if v in value_counts:
                        value_counts[v] += 1
                    else:
                        value_counts[v] = 1
            
            if not value_counts:
                print("No valid colors found in selection")
                return None
                
            # Find the dominant value (the one with the highest frequency)
            dominant_value = max(value_counts, key=value_counts.get)
            print(f"Dominant value (brightness): {dominant_value}")
            
            # Ensure the value is within valid range (0-255)
            if 0 <= dominant_value <= 255:
                return dominant_value
            else:
                print(f"Invalid value extracted: {dominant_value}")
                return None
            
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
                
        if self.isActive and self.parent.fillGroup.isVisible():
            self.selectionTimer.start(500)