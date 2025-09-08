import uuid
import json
from kivy.app import App
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.core.window import Window
from kivy.uix.scrollview import ScrollView
'''
Text Annotation GUI
'''
class ThreeTabsApp(App):
    def build(self):
        Window.maximize()

        panel = TabbedPanel(do_default_tab=False)

        # ---------- Tab 1: Define Labels ----------
        tab1 = TabbedPanelItem(text='Labels')
        layout1 = BoxLayout(orientation='vertical', spacing=10, padding=10)

        input_row = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10)
        self.labels_input = TextInput(hint_text="Enter a label", multiline=False)
        add_btn = Button(text="Add Label", size_hint=(0.3, 1))
        add_btn.bind(on_press=self.add_label)
        input_row.add_widget(self.labels_input)
        input_row.add_widget(add_btn)
        layout1.add_widget(input_row)

        self.labels_grid = GridLayout(cols=3, spacing=5, size_hint=(1, 0.8))
        layout1.add_widget(self.labels_grid)

        tab1.add_widget(layout1)
        panel.add_widget(tab1)

        # ---------- Tab 2: Annotator ----------
        # inside build(), Tab 2 section
        tab2 = TabbedPanelItem(text='Annotator')
        layout2 = BoxLayout(orientation='horizontal', spacing=10, padding=10)

        left_side = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.7, 1))

        file_btn = Button(text="Load Text File", size_hint=(1, 0.1))
        file_btn.bind(on_press=self.open_filechooser)
        left_side.add_widget(file_btn)

        # --- New: jump to line controls ---
        line_jump_row = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=5)
        self.line_input = TextInput(hint_text="Line #", multiline=False, size_hint=(0.7, 1))
        jump_btn = Button(text="Jump", size_hint=(0.3, 1))
        jump_btn.bind(on_press=self.jump_to_line)
        line_jump_row.add_widget(self.line_input)
        line_jump_row.add_widget(jump_btn)
        left_side.add_widget(line_jump_row)

        self.text_box = TextInput(
            text="Highlight part of this text and press 'Assign Label'.",
            multiline=True
        )
        left_side.add_widget(self.text_box)

        assign_btn = Button(text="Assign Label to Selection", size_hint=(1, 0.1))
        assign_btn.bind(on_press=self.assign_label)
        left_side.add_widget(assign_btn)

        layout2.add_widget(left_side)

        right_side = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.3, 1))
        right_side.add_widget(Label(text="Annotations:", size_hint=(1, 0.1)))
        self.label_output = Label(text="(none yet)", size_hint=(1, 0.9))
        right_side.add_widget(self.label_output)

        layout2.add_widget(right_side)
        tab2.add_widget(layout2)
        panel.add_widget(tab2)

        # ---------- Tab 3: Connections ----------
        tab3 = TabbedPanelItem(text='Connections')
        layout3 = BoxLayout(orientation='vertical', spacing=10, padding=10)

        self.connection_status = Label(text="Select two IDs to connect", size_hint=(1, 0.1))
        layout3.add_widget(self.connection_status)

        scroll = ScrollView(size_hint=(1, 0.8))
        self.connection_grid = GridLayout(cols=1, spacing=5, size_hint_y=None)
        self.connection_grid.bind(minimum_height=self.connection_grid.setter('height'))
        scroll.add_widget(self.connection_grid)
        layout3.add_widget(scroll)

        # Export button
        export_btn = Button(text="Export Annotations as JSON", size_hint=(1, 0.1))
        export_btn.bind(on_press=self.export_json)
        layout3.add_widget(export_btn)

        tab3.add_widget(layout3)
        panel.add_widget(tab3)

        # Internal state
        self.available_labels = []
        self.current_label = None
        self.annotations = {}
        self.selected_for_connection = []

        return panel

    def add_label(self, instance):
        label_text = self.labels_input.text.strip()
        if not label_text:
            return
        if label_text in self.available_labels:
            self.labels_input.text = ""
            return

        self.available_labels.append(label_text)
        btn = Button(text=label_text)
        btn.bind(on_press=self.select_label)
        self.labels_grid.add_widget(btn)
        self.labels_input.text = ""

    def select_label(self, instance):
        self.current_label = instance.text

    def assign_label(self, instance):
        selected_text = self.text_box.selection_text
        if not selected_text:
            self.label_output.text = "⚠️ No text selected."
            return
        if not self.current_label:
            self.label_output.text = "⚠️ No label selected. Pick one in Tab 1."
            return

        label_id = uuid.uuid4().hex[:8]
        start = min(self.text_box.selection_from, self.text_box.selection_to)
        stop = max(self.text_box.selection_from, self.text_box.selection_to)

        self.annotations[label_id] = {
            "text": selected_text,
            "label": self.current_label,
            "start": start,
            "stop": stop,
            "connections": []
        }

        self.update_annotations_output()
        self.update_connections_tab()

    def update_annotations_output(self):
        lines = [
            f"[{label_id}] '{data['text']}' → {data['label']} (pos {data['start']}–{data['stop']})"
            for label_id, data in self.annotations.items()
        ]
        self.label_output.text = "\n".join(lines)

    def update_connections_tab(self):
        self.connection_grid.clear_widgets()
        for label_id, data in self.annotations.items():
            display_text = data['text']
            if len(display_text) > 20:
                display_text = display_text[:20] + "..."
            btn = Button(
                text=f"{label_id}: {display_text} ({data['label']})\nConnections: {data['connections']}",
                size_hint_y=None,
                height=60
            )
            btn.bind(on_press=lambda inst, i=label_id: self.pick_for_connection(i))
            self.connection_grid.add_widget(btn)

    def jump_to_line(self, instance):
        """Scroll to a given line number in the text box."""
        try:
            line_num = int(self.line_input.text.strip())
            total_lines = len(self.text_box.text.splitlines())
            if 1 <= line_num <= total_lines:
                # Move cursor to (col=0, row=line_num-1)
                self.text_box.cursor = (0, line_num - 1)
                self.text_box.focus = True  # ensures it scrolls
                self.label_output.text = f"Jumped to line {line_num}"
            else:
                self.label_output.text = f"! Line {line_num} out of range (1–{total_lines})"
        except ValueError:
            self.label_output.text = "! Please enter a valid line number"

    def pick_for_connection(self, label_id):
        if label_id not in self.selected_for_connection:
            self.selected_for_connection.append(label_id)

        if len(self.selected_for_connection) == 2:
            id0, id1 = self.selected_for_connection
            if id1 not in self.annotations[id0]["connections"]:
                self.annotations[id0]["connections"].append(id1)
            if id0 not in self.annotations[id1]["connections"]:
                self.annotations[id1]["connections"].append(id0)

            self.connection_status.text = f"Connected {id0} ↔ {id1}"
            self.selected_for_connection = []
            self.update_connections_tab()

    def open_filechooser(self, instance):
        content = BoxLayout(orientation='vertical', spacing=5)
        chooser = FileChooserListView(filters=["*.txt"])
        load_btn = Button(text="Load Selected", size_hint=(1, 0.2))
        cancel_btn = Button(text="Cancel", size_hint=(1, 0.2))

        popup = Popup(title="Select a Text File", content=content, size_hint=(0.9, 0.9))
        content.add_widget(chooser)
        btn_row = BoxLayout(size_hint=(1, 0.2))
        btn_row.add_widget(load_btn)
        btn_row.add_widget(cancel_btn)
        content.add_widget(btn_row)

        def load_file(_):
            if chooser.selection:
                with open(chooser.selection[0], "r", encoding="utf-8") as f:
                    self.text_box.text = f.read()
                popup.dismiss()

        def cancel(_):
            popup.dismiss()

        load_btn.bind(on_press=load_file)
        cancel_btn.bind(on_press=cancel)

        popup.open()

    def export_json(self, instance):
        """Export annotations to a JSON file."""
        try:
            with open("annotations_export.json", "w", encoding="utf-8") as f:
                json.dump(self.annotations, f, indent=4, ensure_ascii=False)
            self.connection_status.text = "✅ Exported to annotations_export.json"
        except Exception as e:
            self.connection_status.text = f"❌ Export failed: {e}"


if __name__ == "__main__":
    ThreeTabsApp().run()
