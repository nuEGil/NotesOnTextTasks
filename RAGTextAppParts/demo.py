import sys
from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QTextEdit, QPushButton, QFileDialog
)

from TapeReader import GetSentences
def main():
    app = QApplication(sys.argv)

    # main window
    window = QWidget()
    window.setWindowTitle("Text Application")
    window.setGeometry(100, 100, 400, 300)

    # layout
    layout = QVBoxLayout()

    # label - this is the text editing poriton
    label = QLabel("Type something below:")
    layout.addWidget(label)

    # text area
    text_area = QTextEdit()
    text_area.setPlaceholderText("Enter your text here...")
    layout.addWidget(text_area)

    # button
    def open_file():
        file_name, _ = QFileDialog.getOpenFileName(
            window,
            "Open File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_name:
            with open(file_name, "r", encoding="utf-8", errors="ignore") as f:
                text_area.setPlainText(f.read())
            
    button = QPushButton("Upload File")
    button.clicked.connect(open_file)
    layout.addWidget(button)

    # finalize
    window.setLayout(layout)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
