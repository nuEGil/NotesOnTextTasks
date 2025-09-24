import sys
from RAGTextAppParts.sandbox.TapeReader import GetSentences, GetImportantPairsFromList
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFileDialog, 
    QHBoxLayout, QTextEdit, QPushButton, QLabel, 
    QSizePolicy, QSpinBox
)
class MainApplication():
    def __init__(self):
        self.app = QApplication(sys.argv)
        # main window
        self.window = QWidget()
        self.window.setWindowTitle("Text Application")
        self.window.setGeometry(800, 800, 400, 300)

        # Main Layout
        self.MainLayout = QVBoxLayout()

        # label - this is the text editing poriton
        self.label = QLabel("Type something below:")
        self.MainLayout.addWidget(self.label)

        # set a text layout to put the text boxes next to each other
        text_layout = QHBoxLayout()

        ## start with the buttons first so its [button box, text box, text box]
        text_button_layout = QVBoxLayout()
        # Upload file button 
        self.LoadFileButton = QPushButton("Upload File")
        self.LoadFileButton.clicked.connect(self.LoadFileButton_Function) 
        text_button_layout.addWidget(self.LoadFileButton)

        # Get Sentences Button 
        self.GetSentencesButton = QPushButton("Get Sentences")
        self.GetSentencesButton.clicked.connect(self.GetSentencesButton_Function) 
        text_button_layout.addWidget(self.GetSentencesButton)

        # Clear text button 
        self.ResetButton = QPushButton("Reset!")
        self.ResetButton.clicked.connect(self.ResetButton_Function) 
        text_button_layout.addWidget(self.ResetButton)

        text_layout.addLayout(text_button_layout)
        
        ##
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

        # then add the main layout to it 
        self.MainLayout.addLayout(text_layout)

        # going to add 1 more text input so that we can define key names
        self.KeyNameArea = QTextEdit()
        self.KeyNameArea.setPlaceholderText("Enter csv style names -> name,name,name")
        
        self.KeyNameArea.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)

        keyname_area_buttons = QVBoxLayout()
        self.set_keynames = QPushButton("Set")
        self.set_keynames.clicked.connect(self.set_keynames_Function) 
        self.BuildGraph = QPushButton("BuildGraph")
        self.clear_keynames = QPushButton("Clear")
        self.WinSpinBox = QSpinBox()


        keyname_area_buttons.addWidget(self.set_keynames)
        keyname_area_buttons.addWidget(self.BuildGraph)
        keyname_area_buttons.addWidget(self.clear_keynames)
        keyname_area_buttons.addWidget(self.WinSpinBox)

        keyname_area_widgets = QHBoxLayout()
        keyname_area_widgets.addLayout(keyname_area_buttons)
        
        keyname_area_widgets.addWidget(self.KeyNameArea)

        self.MainLayout.addLayout(keyname_area_widgets)

        # finalize
        self.window.setLayout(self.MainLayout)
        self.window.show()

        # create a data structure to record data
        self.data_structure = {'text_input':'',}

    # button press funcitonality
    def LoadFileButton_Function(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self.window, "Open File", 
            "", "Text Files (*.txt);;All Files (*)")

        if file_name:
            with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
                # appending to the text data
                self.data_structure['text_input'] = '' + f.read() 
                self.text_area.setPlainText(self.data_structure['text_input'])

    # button press funcitonality
    def GetSentencesButton_Function(self):
        [starts, ends, sentences] = GetSentences(self.data_structure['text_input'])
        # need to make sure this isnt making that mutable copy of the object. 
        self.data_structure['sentences'] = sentences
        self.data_structure['start_ids'] = starts
        self.data_structure['end_ids'] = ends


        full_text = ''
        for i, sents in enumerate(sentences):
            full_text+=f'\nsentence {i} : {sents}\n...'
        
        self.output_area.setPlainText(full_text)

    def ResetButton_Function(self):
        # clear out the text area and clear out the data store. 
        self.text_area.setPlainText('')
        self.text_area.setPlaceholderText("Enter your text here...")
        
        self.output_area.setPlainText('')
        self.output_area.setPlaceholderText("Processed output will appear here...")
        self.data_structure = {'text_input':'',}
        
    # button press funcitonality
    def set_keynames_Function(self):
        print('Setting key names')
        UserKeyNames = self.KeyNameArea.toPlainText().split(',')
        self.data_structure['UserKeyNames'] = UserKeyNames
        npDC_2, pairs, pairs_counts = GetImportantPairsFromList(UserKeyNames)
        self.data_structure['pairs'] = pairs
        print(pairs)

def main():
    App = MainApplication()
    sys.exit(App.app.exec())

if __name__ == "__main__":
    main()
