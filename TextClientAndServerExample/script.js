const fileInput = document.getElementById("fileInput");
const textArea = document.getElementById("textArea");
const toggles = document.querySelectorAll(".toggle_box input[type='checkbox']");
const clearBtn = document.getElementById("clearBtn");

// clear button functionallity
clearBtn.addEventListener("click", () => {
  textArea.value = "";  // clear the textarea content
});

// File upload 
fileInput.addEventListener("change", () => {
    // define constants that we will use
    const file = fileInput.files[0]; // only one file since no "multiple"
    const allowedExtensions = [".txt", ".csv", ".json"]; // allowed file extensions
    const fileName = file.name.toLowerCase();
    const isValid = allowedExtensions.some(ext => fileName.endsWith(ext));
    
    const reader = new FileReader();

    // check if a file is selected
    if (!file){ 
        alert("No file selected.")
        return;
        }
    // check file extension
    if (!isValid) {
        alert("Invalid file type! Please select a .txt, .csv, or .json file.");
        fileInput.value = ""; // reset the input
        return;
    }

    // logic for when a file is loaded
    reader.onload = (e) => {
    textArea.value = e.target.result; // put file text into textarea
    };

  reader.readAsText(file);
});

// Toggle checking 
toggles.forEach(box => {
  box.addEventListener("change", () => {
    if (box.checked) {
      // uncheck all others
      toggles.forEach(other => {
        if (other !== box) {
          other.checked = false;
        }
      });
    } else {
      // if user tries to uncheck the active one, re-check it
      const anyChecked = Array.from(toggles).some(b => b.checked);
      if (!anyChecked) {
        box.checked = true; // restore this one
      }
    }
  });
});


const parseBtn = document.getElementById("parseBtn");

parseBtn.addEventListener("click", async () => {
  const text = textArea.value.trim();
  if (!text) {
    alert("Please enter or upload some text first.");
    return;
  }

  // find active toggle
  let mode = "hash"; // default
  toggles.forEach(box => {
    if (box.checked) {
      mode = box.id.toLowerCase(); // use the id as mode
    }
  });

//   let encodedText = encodeURIComponent(text).replace(/%20/g, "+");
  try {
    const response = await fetch(`http://127.0.0.1:8000/transform?text=${encodeURIComponent(text)}&mode=${mode}`, {
      method: "POST"
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // put result back in textarea
    textArea.value = data.result;
  } catch (err) {
    console.error(err);
    alert("Error contacting server: " + err.message);
  }
});