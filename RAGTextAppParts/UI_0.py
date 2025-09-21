import re
import sys
from TapeReader import GetSentences
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFileDialog, 
    QHBoxLayout, QTextEdit, QPushButton, QLabel, 
    QSizePolicy, QSpinBox, QMainWindow, QTabWidget, 
)

from PyQt6.QtGui import QTextCharFormat, QColor, QTextCursor
'''Treat this front pannel as sometihg that collects inputs and puts 
them into a data structure. Plan so that 1 button press is a whole pipeline. 
Minimize the number of button presses for a good demo man

Need a way to group buttons together + functinonality. 
'''

class LoadTextPage(QWidget):
    def __init__(self):
        super().__init__()
        # set a text layout to put the text boxes next to each other
        text_layout = QHBoxLayout()

        ## start with the buttons first so its [button box, text box, text box]
        text_button_layout = QVBoxLayout()
        
        # Upload file button 
        self.LoadFileButton = QPushButton("Upload File")
        self.LoadFileButton.clicked.connect(self.LoadFileButton_Function) 
        text_button_layout.addWidget(self.LoadFileButton)

        # Highlight button 
        self.HighlightButton = QPushButton("Highlight")
        self.HighlightButton.clicked.connect(self.HighlightButtonF) 
        text_button_layout.addWidget(self.HighlightButton)

        # Clear text button 
        self.ResetButton = QPushButton("Reset!")
        self.ResetButton.clicked.connect(self.ResetButton_Function) 
        text_button_layout.addWidget(self.ResetButton)

        text_layout.addLayout(text_button_layout)
        
        # Input Text Area
        self.text_area = QTextEdit()
        self.text_area.setPlaceholderText("Enter your text here...")
        self.text_area.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        
        # Output text area (read-only)
        self.output_area = QTextEdit()
        self.output_area.setPlaceholderText("Processed output will appear here...")
        self.output_area.setReadOnly(True)  # prevent editing
        self.output_area.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)

        # add text areas to the layout
        text_layout.addWidget(self.text_area)
        text_layout.addWidget(self.output_area)

        self.setLayout(text_layout)
        # create a data structure to record data
        self.data_structure = {'text_input':'',}

    # button press funcitonality
    def LoadFileButton_Function(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open File", 
            "", "Text Files (*.txt);;All Files (*)")

        if file_name:
            with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
                # appending to the text data
                self.data_structure['text_input'] = '' + f.read() 
                self.text_area.setPlainText(self.data_structure['text_input'])

    def HighlightButtonF(self):
        # cool you know how to highlight things now. 
        cursor = self.text_area.textCursor()
        text_ = self.text_area.toPlainText()
        
        pattern = r"ap"
        for match in re.finditer(pattern, text_):
            cursor.setPosition(match.start())
            cursor.setPosition(match.end(), QTextCursor.MoveMode.KeepAnchor)  
        
            fmt = QTextCharFormat()
            fmt.setBackground(QColor('green'))

            cursor.setCharFormat(fmt)



    def ResetButton_Function(self):
        # clear out the text area and clear out the data store. 
        self.text_area.setPlainText('')
        self.text_area.setPlaceholderText("Enter your text here...")
        
        self.output_area.setPlainText('')
        self.output_area.setPlaceholderText("Processed output will appear here...")
        self.data_structure = {'text_input':'',}

class SettingsPage(QWidget):
    def __init__(self):
        super().__init__()
        # going to add 1 more text input so that we can define key names
        self.KeyNameArea = QTextEdit()
        self.KeyNameArea.setPlaceholderText("Enter csv style names -> name,name,name")
        
        self.KeyNameArea.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)

        # vertical layout for the buttons and spin box
        keyname_area_buttons = QVBoxLayout()
        self.set_keynames = QPushButton("Set")
        # self.set_keynames.clicked.connect(self.set_keynames_Function) 
        self.BuildGraph = QPushButton("BuildGraph")
        self.clear_keynames = QPushButton("Clear")
        self.WinSpinBox = QSpinBox()
        self.WinSpinBox.setValue(3)

        # add in widgets. 
        keyname_area_buttons.addWidget(self.set_keynames)
        keyname_area_buttons.addWidget(self.BuildGraph)
        keyname_area_buttons.addWidget(self.clear_keynames)

        keyname_area_buttons.addWidget(QLabel("Sentence Window Length:"))
        keyname_area_buttons.addWidget(self.WinSpinBox)

        keyname_area_widgets = QHBoxLayout()
        keyname_area_widgets.addLayout(keyname_area_buttons)
        
        keyname_area_widgets.addWidget(self.KeyNameArea)

        self.setLayout(keyname_area_widgets)

class TabbedApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tabbed PyQt6 App")
        self.setGeometry(200, 200, 700, 500)

        # Create Tab Widget
        tabs = QTabWidget()

        # Add tabs 1 and 2. 
        tabs.addTab(SettingsPage(), "Settings Page")
        tabs.addTab(LoadTextPage(), "Text File Viewer")
        

        # Continue adding tabs. -- remember they just have to be widgets. 
        second_page = QWidget()
        second_layout = QVBoxLayout()
        second_layout.addWidget(QLabel("This is another page!"))
        second_page.setLayout(second_layout)
        tabs.addTab(second_page, "Second Page")

        self.setCentralWidget(tabs)

def main():
    # can the main app be a collection of classes?
    app = QApplication(sys.argv)
    window = TabbedApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()