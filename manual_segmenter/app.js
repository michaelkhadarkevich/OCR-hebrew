const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const chooseImagesBtn = document.getElementById("chooseImagesBtn");
const chooseFolderBtn = document.getElementById("chooseFolderBtn");
const pageSelect = document.getElementById("pageSelect");
const pageInfo = document.getElementById("pageInfo");
const transcription = document.getElementById("transcription");
const deleteBtn = document.getElementById("deleteBtn");
const toggleBoxTextBtn = document.getElementById("toggleBoxTextBtn");
const savePageBtn = document.getElementById("savePageBtn");
const saveAllBtn = document.getElementById("saveAllBtn");
const outputPath = document.getElementById("outputPath");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const zoomInput = document.getElementById("zoomInput");
const statusEl = document.getElementById("status");
const boxList = document.getElementById("boxList");
const helpBtn = document.getElementById("helpBtn");
const helpPanel = document.getElementById("helpPanel");
const canvas = document.getElementById("pageCanvas");
const canvasWrap = document.getElementById("canvasWrap");
const ctx = canvas.getContext("2d");

let pages = [];
let currentPageIndex = -1;
let selectedBoxId = null;
let drawing = null;
let resizing = null;
let nextBoxNumber = 1;
let showBoxText = true;
const MIN_BOX_WIDTH = 20;
const MIN_BOX_HEIGHT = 20;
const HANDLE_SIZE = 10;

function setStatus(message) {
  statusEl.textContent = message;
}

function safeBaseName(name) {
  return name.replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "_");
}

function currentPage() {
  return pages[currentPageIndex] || null;
}

function drawBoxText(box, isSelected) {
  if (!showBoxText || !box.text) return;
  const text = box.text;
  const fontSize = Math.max(12, Math.min(20, Math.floor(box.height * 0.45)));
  ctx.font = `${fontSize}px Arial`;
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  const x = box.x + box.width - 4;
  const y = box.y + 4;
  const metrics = ctx.measureText(text);
  const bgWidth = Math.min(box.width - 4, metrics.width + 8);
  const bgHeight = fontSize + 6;
  ctx.fillStyle = isSelected ? "rgba(15, 118, 110, 0.85)" : "rgba(255, 255, 255, 0.82)";
  ctx.fillRect(Math.max(box.x + 2, x - bgWidth), y - 2, bgWidth, bgHeight);
  ctx.fillStyle = isSelected ? "white" : "#111827";
  ctx.fillText(text, x, y);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}
function boxHandles(box) {
  const s = HANDLE_SIZE;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  const x2 = box.x + box.width;
  const y2 = box.y + box.height;
  return [
    { name: "nw", x: box.x, y: box.y, cursor: "nwse-resize" },
    { name: "n", x: cx, y: box.y, cursor: "ns-resize" },
    { name: "ne", x: x2, y: box.y, cursor: "nesw-resize" },
    { name: "e", x: x2, y: cy, cursor: "ew-resize" },
    { name: "se", x: x2, y: y2, cursor: "nwse-resize" },
    { name: "s", x: cx, y: y2, cursor: "ns-resize" },
    { name: "sw", x: box.x, y: y2, cursor: "nesw-resize" },
    { name: "w", x: box.x, y: cy, cursor: "ew-resize" },
  ].map((handle) => ({ ...handle, left: handle.x - s / 2, top: handle.y - s / 2, size: s }));
}

function drawResizeHandles(box) {
  ctx.save();
  for (const handle of boxHandles(box)) {
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 2;
    ctx.fillRect(handle.left, handle.top, handle.size, handle.size);
    ctx.strokeRect(handle.left, handle.top, handle.size, handle.size);
  }
  ctx.restore();
}

function handleAtPoint(box, point) {
  if (!box) return null;
  return boxHandles(box).find((handle) =>
    point.x >= handle.left && point.x <= handle.left + handle.size &&
    point.y >= handle.top && point.y <= handle.top + handle.size
  ) || null;
}

function selectedBox(page) {
  return page?.boxes.find((box) => box.id === selectedBoxId) || null;
}

function resizedRect(startBox, handleName, point) {
  let x1 = startBox.x;
  let y1 = startBox.y;
  let x2 = startBox.x + startBox.width;
  let y2 = startBox.y + startBox.height;
  if (handleName.includes("w")) x1 = point.x;
  if (handleName.includes("e")) x2 = point.x;
  if (handleName.includes("n")) y1 = point.y;
  if (handleName.includes("s")) y2 = point.y;
  if (x2 - x1 < MIN_BOX_WIDTH) {
    if (handleName.includes("w")) x1 = x2 - MIN_BOX_WIDTH;
    else x2 = x1 + MIN_BOX_WIDTH;
  }
  if (y2 - y1 < MIN_BOX_HEIGHT) {
    if (handleName.includes("n")) y1 = y2 - MIN_BOX_HEIGHT;
    else y2 = y1 + MIN_BOX_HEIGHT;
  }
  x1 = Math.max(0, Math.min(canvas.width - MIN_BOX_WIDTH, x1));
  y1 = Math.max(0, Math.min(canvas.height - MIN_BOX_HEIGHT, y1));
  x2 = Math.max(x1 + MIN_BOX_WIDTH, Math.min(canvas.width, x2));
  y2 = Math.max(y1 + MIN_BOX_HEIGHT, Math.min(canvas.height, y2));
  return { x: Math.round(x1), y: Math.round(y1), width: Math.round(x2 - x1), height: Math.round(y2 - y1) };
}

function redraw() {
  const page = currentPage();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!page) return;
  ctx.drawImage(page.image, 0, 0);
  renumberBoxesByPosition(page);

  for (const box of page.boxes) {
    ctx.lineWidth = selectedBoxId === box.id ? 4 : 2;
    ctx.strokeStyle = selectedBoxId === box.id ? "#0f766e" : "#dc2626";
    ctx.strokeRect(box.x, box.y, box.width, box.height);
    ctx.fillStyle = selectedBoxId === box.id ? "#0f766e" : "#dc2626";
    ctx.font = "18px Arial";
    ctx.fillText(String(box.order), box.x + 4, Math.max(20, box.y - 6));
    drawBoxText(box, selectedBoxId === box.id);
  }

  if (drawing) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#2563eb";
    ctx.setLineDash([8, 5]);
    ctx.strokeRect(drawing.x, drawing.y, drawing.width, drawing.height);
    ctx.setLineDash([]);
  }
}

function sortedBoxesByPosition(boxes) {
  return [...boxes].sort((a, b) => {
    const ay = a.y + a.height / 2;
    const by = b.y + b.height / 2;
    const lineTolerance = Math.max(10, Math.min(a.height, b.height) * 0.6);
    if (Math.abs(ay - by) > lineTolerance) return ay - by;
    const ax = a.x + a.width / 2;
    const bx = b.x + b.width / 2;
    return bx - ax; // Hebrew/RTL: right side first when boxes are on the same row.
  });
}

function renumberBoxesByPosition(page) {
  if (!page) return [];
  page.boxes = sortedBoxesByPosition(page.boxes);
  page.boxes.forEach((box, index) => {
    box.order = index + 1;
  });
  return page.boxes;
}
function renderBoxList() {
  const page = currentPage();
  boxList.innerHTML = "";
  if (!page) return;
  renumberBoxesByPosition(page);
  for (const box of page.boxes) {
    const li = document.createElement("li");
    li.textContent = `${box.order}. ${box.text || "(no text)"}`;
    if (box.id === selectedBoxId) li.classList.add("selected");
    li.addEventListener("click", () => selectBox(box.id));
    boxList.appendChild(li);
  }
}

function updatePageSelect() {
  pageSelect.innerHTML = "";
  pages.forEach((page, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${index + 1}. ${page.file.name}`;
    pageSelect.appendChild(option);
  });
  pageSelect.value = String(currentPageIndex);
}

function scrollBoxIntoView(box, force = false) {
  if (!box || !canvasWrap) return;
  const scale = Number(zoomInput.value) / 100;
  const margin = 40;
  const left = box.x * scale;
  const top = box.y * scale;
  const right = (box.x + box.width) * scale;
  const bottom = (box.y + box.height) * scale;

  if (force) {
    canvasWrap.scrollLeft = Math.max(0, left - margin);
    canvasWrap.scrollTop = Math.max(0, top - margin);
    // If the browser page, not the image panel, is currently the scroller, keep the canvas visible too.
    canvas.scrollIntoView({ block: "nearest", inline: "nearest" });
    return;
  }

  const viewLeft = canvasWrap.scrollLeft;
  const viewTop = canvasWrap.scrollTop;
  const viewRight = viewLeft + canvasWrap.clientWidth;
  const viewBottom = viewTop + canvasWrap.clientHeight;

  if (left < viewLeft + margin) {
    canvasWrap.scrollLeft = Math.max(0, left - margin);
  } else if (right > viewRight - margin) {
    canvasWrap.scrollLeft = Math.max(0, right - canvasWrap.clientWidth + margin);
  }
  if (top < viewTop + margin) {
    canvasWrap.scrollTop = Math.max(0, top - margin);
  } else if (bottom > viewBottom - margin) {
    canvasWrap.scrollTop = Math.max(0, bottom - canvasWrap.clientHeight + margin);
  }
}

function scrollSelectedBoxIntoView() {
  const page = currentPage();
  scrollBoxIntoView(selectedBox(page));
}
function focusTranscription(selectText = true) {
  transcription.disabled = false;
  setTimeout(() => {
    transcription.focus({ preventScroll: true });
    if (selectText) transcription.select();
  }, 0);
}

function selectBox(id) {
  const page = currentPage();
  const hadSelectedBox = Boolean(selectedBoxId);
  const pendingText = transcription.value.trim();
  const box = page?.boxes.find((item) => item.id === id);

  if (box && !hadSelectedBox && !box.text && pendingText) {
    box.text = pendingText;
  }

  selectedBoxId = box ? id : null;
  transcription.value = box ? box.text : "";
  renderBoxList();
  redraw();
  if (box) {
    focusTranscription(!(pendingText && box.text === pendingText));
    setTimeout(() => scrollSelectedBoxIntoView(), 0);
  }
}

function render() {
  const page = currentPage();
  const hasPage = Boolean(page);
  pageInfo.textContent = hasPage ? `${page.image.width} x ${page.image.height}, boxes: ${page.boxes.length}` : "No page loaded.";
  transcription.disabled = false;
  deleteBtn.disabled = !hasPage || !selectedBoxId;
  savePageBtn.disabled = !hasPage;
  saveAllBtn.disabled = !pages.length;
  prevBtn.disabled = currentPageIndex <= 0;
  nextBtn.disabled = currentPageIndex < 0 || currentPageIndex >= pages.length - 1;

  if (page) {
    canvas.width = page.image.width;
    canvas.height = page.image.height;
  } else {
    canvas.width = 1;
    canvas.height = 1;
  }
  canvas.style.transform = `scale(${Number(zoomInput.value) / 100})`;
  canvas.style.transformOrigin = "top left";
  renderBoxList();
  redraw();
}

function isImageFile(file) {
  return file.type.startsWith("image/") || /\.(png|jpe?g|webp|bmp)$/i.test(file.name);
}

async function loadImageFile(file) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = url;
    });
    return { file, image, boxes: [], baseName: safeBaseName(file.name) };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function loadFiles(files) {
  pages = [];
  selectedBoxId = null;
  nextBoxNumber = 1;
  for (const file of files) pages.push(await loadImageFile(file));
  currentPageIndex = pages.length ? 0 : -1;
  updatePageSelect();
render();
  setStatus(pages.length ? `Loaded ${pages.length} page image(s).` : "No image files loaded.");
}

chooseImagesBtn.addEventListener("click", () => {
  fileInput.value = "";
  fileInput.click();
});

chooseFolderBtn.addEventListener("click", () => {
  folderInput.value = "";
  folderInput.click();
});

fileInput.addEventListener("change", async () => {
  await loadFiles([...fileInput.files].filter(isImageFile));
});

folderInput.addEventListener("change", async () => {
  const files = [...folderInput.files]
    .filter(isImageFile)
    .sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name, undefined, { numeric: true }));
  await loadFiles(files);
});

pageSelect.addEventListener("change", () => {
  currentPageIndex = Number(pageSelect.value);
  selectedBoxId = null;
  transcription.value = "";
render();
});

prevBtn.addEventListener("click", () => {
  if (currentPageIndex > 0) {
    currentPageIndex--;
    selectedBoxId = null;
    transcription.value = "";
    updatePageSelect();
render();
  }
});

nextBtn.addEventListener("click", () => {
  if (currentPageIndex < pages.length - 1) {
    currentPageIndex++;
    selectedBoxId = null;
    transcription.value = "";
    updatePageSelect();
render();
  }
});

zoomInput.addEventListener("input", render);

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return { x: Math.round((event.clientX - rect.left) * scaleX), y: Math.round((event.clientY - rect.top) * scaleY) };
}

function boxAtPoint(page, point) {
  if (!page) return null;
  for (let i = page.boxes.length - 1; i >= 0; i--) {
    const box = page.boxes[i];
    if (
      point.x >= box.x &&
      point.x <= box.x + box.width &&
      point.y >= box.y &&
      point.y <= box.y + box.height
    ) {
      return box;
    }
  }
  return null;
}

canvas.addEventListener("contextmenu", (event) => event.preventDefault());

canvas.addEventListener("mousedown", (event) => {
  const page = currentPage();
  if (!page) return;
  const p = canvasPoint(event);
  const activeBox = selectedBox(page);
  const activeHandle = handleAtPoint(activeBox, p);
  const clickedBox = boxAtPoint(page, p);

  if (event.button === 0) {
    if (activeBox && activeHandle) {
      resizing = { boxId: activeBox.id, handle: activeHandle.name, startBox: { ...activeBox } };
      canvas.style.cursor = activeHandle.cursor;
      setStatus(`Resizing box ${activeBox.order}.`);
      return;
    }
    if (clickedBox) {
      selectBox(clickedBox.id);
      setStatus(`Selected box ${clickedBox.order}. Drag a small square handle to resize.`);
      return;
    }
    drawing = { startX: p.x, startY: p.y, x: p.x, y: p.y, width: 0, height: 0 };
    return;
  }

  if (event.button === 2) {
    if (clickedBox) selectedBoxId = clickedBox.id;
    deleteSelectedBox();
    return;
  }
});

canvas.addEventListener("mousemove", (event) => {
  const page = currentPage();
  if (!page) return;
  const p = canvasPoint(event);

  if (resizing) {
    const box = page.boxes.find((item) => item.id === resizing.boxId);
    if (!box) return;
    Object.assign(box, resizedRect(resizing.startBox, resizing.handle, p));
    renderBoxList();
    redraw();
    return;
  }

  if (drawing) {
    drawing.x = Math.min(drawing.startX, p.x);
    drawing.y = Math.min(drawing.startY, p.y);
    drawing.width = Math.abs(p.x - drawing.startX);
    drawing.height = Math.abs(p.y - drawing.startY);
    redraw();
    return;
  }

  const handle = handleAtPoint(selectedBox(page), p);
  canvas.style.cursor = handle ? handle.cursor : "crosshair";
});

function finishResize() {
  const page = currentPage();
  if (!resizing) return false;
  const box = page?.boxes.find((item) => item.id === resizing.boxId);
  resizing = null;
  canvas.style.cursor = "crosshair";
  if (box) setStatus(`Resized box ${box.order}.`);
  redraw();
  return true;
}
canvas.addEventListener("mouseleave", () => {
  if (drawing) {
    drawing = null;
    redraw();
  }
  finishResize();
});

canvas.addEventListener("mouseup", () => {
  const page = currentPage();
  if (finishResize()) return;
  if (!page || !drawing) return;
  const rect = drawing;
  drawing = null;
  if (rect.width < MIN_BOX_WIDTH || rect.height < MIN_BOX_HEIGHT) {
    redraw();
    setStatus(`Box too small. Minimum is ${MIN_BOX_WIDTH} x ${MIN_BOX_HEIGHT} pixels.`);
    return;
  }
  const box = { id: crypto.randomUUID(), order: nextBoxNumber++, x: rect.x, y: rect.y, width: rect.width, height: rect.height, text: "" };
  page.boxes.push(box);
  selectBox(box.id);
  setStatus(`Added box ${box.order}. Type transcription; it auto-saves.`);
});

transcription.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  event.preventDefault();
  selectNextBox();
});
transcription.addEventListener("input", () => {
  const page = currentPage();
  const box = page?.boxes.find((item) => item.id === selectedBoxId);
  if (!box) return;
  box.text = transcription.value.trim();
  renderBoxList();
  redraw();
  scrollSelectedBoxIntoView();
  setStatus(`Auto-saved text for box ${box.order}.`);
});

function deleteSelectedBox() {
  const page = currentPage();
  if (!page || !selectedBoxId) return false;
  const box = page.boxes.find((item) => item.id === selectedBoxId);
  page.boxes = page.boxes.filter((item) => item.id !== selectedBoxId);
  selectedBoxId = null;
  transcription.value = "";
render();
  setStatus(box ? `Deleted box ${box.order}.` : "Deleted selected box.");
  return true;
}

function selectNextBox() {
  const page = currentPage();
  if (!page || !page.boxes.length) return false;
  renumberBoxesByPosition(page);
  const index = page.boxes.findIndex((box) => box.id === selectedBoxId);
  const nextIndex = index >= 0 ? Math.min(index + 1, page.boxes.length - 1) : 0;
  const nextBox = page.boxes[nextIndex];
  if (!nextBox || nextBox.id === selectedBoxId) return false;
  selectBox(nextBox.id);
  setTimeout(() => scrollBoxIntoView(nextBox, true), 0);
  setTimeout(() => scrollBoxIntoView(nextBox, true), 80);
  setStatus(`Selected box ${nextBox.order}.`);
  return true;
}
document.addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  if (deleteSelectedBox()) {
    event.preventDefault();
  }
});
deleteBtn.addEventListener("click", () => {
  deleteSelectedBox();
});

function cropToBlob(page, box) {
  const crop = document.createElement("canvas");
  crop.width = box.width;
  crop.height = box.height;
  const cropCtx = crop.getContext("2d");
  cropCtx.drawImage(page.image, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height);
  return new Promise((resolve) => crop.toBlob(resolve, "image/png"));
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function saveFilesToServer(outputFolder, files) {
  const response = await fetch("/save_page", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ output_folder: outputFolder, files }),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

async function savePage(page) {
  if (!page) return setStatus("No page loaded.");
  if (!page.boxes.length) return setStatus("No boxes on this page to save.");
  const out = outputPath.value.trim();
  if (!out) return setStatus("Please type an output folder first.");

  const files = [];
  renumberBoxesByPosition(page);
  for (const box of page.boxes) {
    const stem = `${page.baseName}_seg_${String(box.order).padStart(4, "0")}`;
    const blob = await cropToBlob(page, box);
    files.push({ name: `${stem}.png`, kind: "base64", data_url: await blobToDataUrl(blob) });
    files.push({ name: `${stem}.txt`, kind: "text", text: box.text + "\n" });
  }
  const result = await saveFilesToServer(out, files);
  setStatus(`Saved ${page.boxes.length} crop(s) to ${result.output_folder}.`);
}

savePageBtn.addEventListener("click", async () => {
  try {
    await savePage(currentPage());
  } catch (error) {
    setStatus(`Save failed: ${error.message}`);
    console.error(error);
  }
});

saveAllBtn.addEventListener("click", async () => {
  try {
    for (const page of pages) await savePage(page);
    setStatus(`Saved all pages: ${pages.reduce((sum, page) => sum + page.boxes.length, 0)} crop(s).`);
  } catch (error) {
    setStatus(`Save failed: ${error.message}`);
    console.error(error);
  }
});
toggleBoxTextBtn.addEventListener("click", () => {
  showBoxText = !showBoxText;
  toggleBoxTextBtn.textContent = showBoxText ? "Hide text in boxes" : "Show text in boxes";
  redraw();
});
helpBtn.addEventListener("click", () => {
  helpPanel.hidden = !helpPanel.hidden;
});
render();
