//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// https://emscripten.org/docs/api_reference/module.html
var Module;


function print(text) {
  text = Array.prototype.slice.call(arguments).join(' ')
  console.log(text)
}

function printErr(text) {
  text = Array.prototype.slice.call(arguments).join(' ')
  console.error(text)
}


function setStatus(text) {
  // console.log('status: ' + text)
}

function monitorRunDependencies(left) {
  // no run dependencies to log
}

function stripJsonComments(jsonText) {
  // Split the text into lines, filter out comment lines, and rejoin
  return jsonText
    .split('\n')
    .filter(line => !line.trim().startsWith('//'))
    .join('\n');
}

function canvasDropHandler(e) {
  e.preventDefault()
  const files = e.dataTransfer.files
  if (files.length > 0) {
    const file = files[0]
    if (file.type === 'application/json') {
      const reader = new FileReader()
      reader.onload = function (event) {
        const jsonData = event.target.result
        const cleanJsonData = stripJsonComments(jsonData)
        const jsonstring = stringToNewUTF8(cleanJsonData)
        Module.ccall('handleJsonData', 'void', ['number'], [jsonstring])
      }
      reader.readAsText(file)
    }
    if (file.type === 'application/x-yaml') {
      const reader = new FileReader()
      reader.onload = function (event) {
        const yamlutf8 = stringToNewUTF8(event.target.result)
        Module.ccall('handleYamlData', 'void', ['number'], [yamlutf8])
      }
      reader.readAsText(file)
    }
  }
}

function getCanvas() {
  var canvas = document.getElementById('main-render-window-canvas');
  canvas.addEventListener('dragover', (e) => e.preventDefault());
  canvas.addEventListener('drop', canvasDropHandler);
  return canvas;
}

async function initWebGPUDevice() {
  if (!navigator.gpu) {
    throw Error('WebGPU not supported.')
  }

  const adapter = await navigator.gpu.requestAdapter()
  const device = await adapter.requestDevice()
  return device;
}

async function createEmscriptenModule() {
  // https://emscripten.org/docs/api_reference/module.html
  return {
    preRun: [],
    postRun: [],
    print: print,
    printErr: printErr,
    canvas: getCanvas(),
    preinitializedWebGPUDevice: await initWebGPUDevice(),
    setStatus: setStatus,
    monitorRunDependencies: monitorRunDependencies,
  }
}

function setEditorContent(content) {
  const editor = document.querySelector('#json-editor')
  editor.textContent = content
}

function loadJsonData(fileName) {
  if (fileName.length === 0) {
    return
  }
  const storedData = localStorage.getItem(fileName)
  if (storedData) {
    setEditorContent(storedData)
  } else {
    fetch(`layouts/${fileName}`)
      .then(response => {
        if (response.ok) return response.text()
        throw new Error(`Error fetching ${fileName}: ${response.statusText}`)
      })
      .then(data => {
        setEditorContent(data)
      })
      .catch(error => console.error(error))
  }
}

function fetchFilesList() {
  return fetch('layouts/files.txt')
    .then(response => {
      if (response.ok) return response.text()
      throw new Error(`Error fetching files.txt: ${response.statusText}`)
    })
    .then(data => {
      // Split by line, filter out empty lines and lines starting with '#'
      return data.split('\n').filter(filename => filename.trim().length > 0 && !filename.trim().startsWith('#'))
    })
    .catch(error => {
      console.error(error)
      return [] // Return empty array on error
    })
}

function sendJsonToWasm() {
  const editor = document.querySelector('#json-editor')
  const rawJsonData = editor.innerText;
  const reformattedJsonData = reformatJSON(rawJsonData);
  if (reformattedJsonData === null) {
    return;
  }
  editor.innerText = reformattedJsonData;
  const cleanJsonData = stripJsonComments(rawJsonData);
  const jsonString = stringToNewUTF8(cleanJsonData)
  Module.ccall('handleLayoutJson', 'void', ['number'], [jsonString])
  _free(jsonString)
}

function loadAllFilesInOrder(filesList) {
  const promises = filesList.map(fileName => {
    return fetch(`layouts/${fileName}`)
      .then(response => {
        if (response.ok) return response.text()
        throw new Error(`Error fetching ${fileName}: ${response.statusText}`)
      })
      .then(data => {
        //console.log(`Loaded ${fileName}:`, data)
        // Store the data in localStorage or a global variable
        localStorage.setItem(fileName, data)
      })
      .catch(error => console.error(error))
  })

  // Wait for all files to be loaded
  return Promise.all(promises)
}

function setupLoadAllButton() {
  const loadAllButton = document.getElementById('load-all')
  loadAllButton.addEventListener('click', function () {
    fetchFilesList()
      .then(filesList => {
        return loadAllFilesInOrder(filesList)
      })
      .then(() => {
        const keys = Object.keys(localStorage)
        // Ensure we process files in the order they appear in filesList
        fetchFilesList().then(filesList => {
          filesList.forEach(fileName => {
            if (keys.includes(fileName)) {
              const jsonData = localStorage.getItem(fileName)
              if (jsonData) {
                sendJsonToWasm(jsonData)
              }
            }
          })
        })
      })
      .then(() => {
        alert('All JSON files have been sent to the WASM module in order.')
      })
      .catch(error => console.error(error))
  })
}

function populateLayoutSelector() {
  const selector = document.getElementById('layout-selector')
  selector.innerHTML = '' // Clear existing options

  // First try to fetch from server file list
  fetchFilesList()
    .then(filesList => {
      if (filesList.length > 0) {
        // Use files from the server list
        filesList.forEach(filename => {
          const option = document.createElement('option')
          option.value = filename
          option.textContent = filename
          selector.appendChild(option)
        })

        // Load all files in order
        return loadAllFilesInOrder(filesList)
      } else {
        // Fall back to local storage if server list is empty
        const keys = Object.keys(localStorage)
        keys.forEach(key => {
          if (key.endsWith('.json')) {
            const option = document.createElement('option')
            option.value = key
            option.textContent = key
            selector.appendChild(option)
          }
        })
      }
    })
    .then(() => {
      // Trigger loading the selected file
      const selectedFile = selector.value
      if (selectedFile) loadJsonData(selectedFile)
    })
    .catch(error => console.error(error))
}

function setupLayoutSelector() {
  populateLayoutSelector() // Populate the dropdown initially
  const selector = document.getElementById('layout-selector')
  selector.addEventListener('change', function () {
    const selectedFile = selector.value
    loadJsonData(selectedFile)
  })
}

function setupSyncToWasm() {
  const editor = document.querySelector('#json-editor')
  editor.addEventListener('keydown', function (event) {
    if (event.ctrlKey && event.key === 'Enter') {
      event.preventDefault()
      sendJsonToWasm();

      const selectedFile = document.getElementById('layout-selector').value
      localStorage.setItem(selectedFile, editor.textContent)
    }
  })
}

function callLoadJsonData() {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      const defaultFile = document.getElementById('layout-selector').value
      loadJsonData(defaultFile)
    })
  } else {
    const defaultFile = document.getElementById('layout-selector').value
    loadJsonData(defaultFile)
  }
}

async function initEmscriptenModuleAndLoadWASM() {
  Module = await createEmscriptenModule();

  window.onerror = function (event) {
    console.log('onerror: ' + event)
  }

  {
    const js = document.createElement('script')
    js.async = true
    js.src = 'motr_gui.js'
    document.body.appendChild(js)
  }
}


function setupClearStorageButton() {
  const clearButton = document.getElementById('clear-storage')
  clearButton.addEventListener('click', function () {
    const keys = Object.keys(localStorage)
    keys.forEach(key => {
      if (key.endsWith('.json')) {
        localStorage.removeItem(key)
      }
    })
    alert('Local storage for layout files cleared.')
  })
}

function setupCreateNewFileButton() {
  const createButton = document.getElementById('create-new-file')
  const newFileNameInput = document.getElementById('new-file-name')

  createButton.addEventListener('click', function () {
    const newFileName = newFileNameInput.value.trim()
    if (newFileName && !newFileName.endsWith('.json')) {
      alert('File name must end with .json')
      return
    }
    if (newFileName) {
      localStorage.setItem(newFileName, '')
      alert(`New file ${newFileName} created in local storage.`)
      newFileNameInput.value = '' // Clear the input field
    } else {
      alert('Please enter a valid file name.')
    }
  })
}

function showErrorContext(jsonText, error) {
  const regex = /at position \d+ \(line (\d+) column (\d+)\)/;
  const match = error.message.match(regex);

  if (match) {
    const errorLine = parseInt(match[1], 10) - 1; // Convert to zero-based index
    const errorCol = parseInt(match[2], 10) - 1; // Convert to zero-based index

    const lines = jsonText.split('\n');
    const startLine = Math.max(0, errorLine - 3); // Get three lines before the error
    const endLine = errorLine + 1; // Include the error line

    const contextLines = lines.slice(startLine, endLine);
    const errorPointer = ' '.repeat(errorCol) + '^'; // Create a caret pointing to the error

    // Construct the error message
    const errorMessage = contextLines.join('\n') + '\n' + errorPointer;

    console.error(`Error at line=${match[1]}, col=${match[2]}:\n${errorMessage}`);
    alert(`Error at line=${match[1]}, col=${match[2]}:\n${errorMessage}`);
  } else {
    console.error('Failed to format JSON:', error.message);
    alert('Failed to format JSON');
  }
}

function reformatJSON(jsonText) {
  const lines = jsonText.split('\n');
  const commentLines = [];
  const jsonLines = [];

  // Separate comments and JSON lines
  lines.forEach((line, index) => {
    if (line.trim().startsWith('//')) {
      commentLines.push({ line: index, text: line });
    } else {
      jsonLines.push(line);
    }
  });

  // Join JSON lines and parse
  const nonCommentText = jsonLines.join('\n');
  let formattedJson;
  try {
    const jsonObj = JSON.parse(nonCommentText);
    formattedJson = JSON.stringify(jsonObj, null, 2);
  } catch (error) {
    showErrorContext(jsonText, error);
    return null; // Return null to indicate failure
  }

  // Reinsert comments at their original positions
  const formattedLines = formattedJson.split('\n');
  commentLines.forEach(comment => {
    formattedLines.splice(comment.line, 0, comment.text);
  });

  return formattedLines.join('\n');
}

function doJSONReformat() {
  const editor = document.querySelector('#json-editor');
  const formatted = reformatJSON(editor.textContent);
  if (formatted !== null) {
    editor.textContent = formatted;
  }
}

function setupReformatButton() {
  const reformatButton = document.getElementById('reformat-code')
  reformatButton.addEventListener('click', doJSONReformat)
}

function checkEditParameter() {
  const urlParams = new URLSearchParams(window.location.search)
  const editMode = urlParams.get('edit') === 'true'
  const editorContainer = document.getElementById('editor-container')

  if (editMode) {
    editorContainer.style.display = 'flex'
  } else {
    editorContainer.style.display = 'none'
  }
}

document.addEventListener('keydown', function(event) {
  if (event.key === 'F12') {
    // Allow the default behavior (open developer tools)
    event.stopPropagation();
  }

  // Check for Cmd-R (Mac) or Ctrl-R (Windows/Linux)
  if ((event.metaKey && event.key === 'r') ||
      (event.ctrlKey && event.key === 'r')) {
    // Allow the default behavior (reload page)
    event.stopPropagation();
  }

}, true)

function downloadText(text, basename, extension) {
  // Define a mapping of file extensions to MIME types
  const mimeTypes = {
    txt: 'text/plain',
    csv: 'text/csv',
    tsv: 'text/tab-separated-values'
  };

  // Get the MIME type based on the extension, default to 'text/plain' if not found
  const mimeType = mimeTypes[extension] || 'text/plain';

  // Create a Blob with the specified MIME type
  const blob = new Blob([text], { type: mimeType });

  // Create a URL for the Blob
  const url = URL.createObjectURL(blob);

  // Create an anchor element and trigger a download
  const a = document.createElement('a');
  a.href = url;
  a.download = `${basename}.${extension}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);

  // Revoke the object URL to free up resources
  URL.revokeObjectURL(url);
}

initEmscriptenModuleAndLoadWASM();
checkEditParameter();
setupLayoutSelector();
setupSyncToWasm();
callLoadJsonData();
setupClearStorageButton()
setupCreateNewFileButton()
setupReformatButton()
setupLoadAllButton()