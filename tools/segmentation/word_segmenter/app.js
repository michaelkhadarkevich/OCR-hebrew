const pdfInput = document.getElementById("pdfInput");
const fileInput = document.getElementById("fileInput");
const datasetFolderInput = document.getElementById("datasetFolderInput");
const choosePdfBtn = document.getElementById("choosePdfBtn");
const chooseImagesBtn = document.getElementById("chooseImagesBtn");
const chooseDatasetFolderBtn = document.getElementById("chooseDatasetFolderBtn");
const pageSelect = document.getElementById("pageSelect");
const pageInfo = document.getElementById("pageInfo");
const sourceTranscription = document.getElementById("sourceTranscription");
const assignWordsBtn = document.getElementById("assignWordsBtn");
const wordCountInfo = document.getElementById("wordCountInfo");
const autoAfterLoad = document.getElementById("autoAfterLoad");
const inkThresholdInput = document.getElementById("inkThresholdInput");
const gapInkMaxInput = document.getElementById("gapInkMaxInput");
const minGapInput = document.getElementById("minGapInput");
const minWordWidthInput = document.getElementById("minWordWidthInput");
const autoCurrentBtn = document.getElementById("autoCurrentBtn");
const autoAllBtn = document.getElementById("autoAllBtn");
const clearBoxesBtn = document.getElementById("clearBoxesBtn");
const autoInfo = document.getElementById("autoInfo");
const transcription = document.getElementById("transcription");
const deleteBtn = document.getElementById("deleteBtn");
const toggleBoxTextBtn = document.getElementById("toggleBoxTextBtn");
const savePageBtn = document.getElementById("savePageBtn");
const saveNextBtn = document.getElementById("saveNextBtn");
const saveAllBtn = document.getElementById("saveAllBtn");
const outputPath = document.getElementById("outputPath");
const paddingInput = document.getElementById("paddingInput");
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
let showBoxText = true;
const MIN_BOX_WIDTH = 12;
const MIN_BOX_HEIGHT = 12;
const HANDLE_SIZE = 10;

function setStatus(message) { statusEl.textContent = message; }
function safeBaseName(name) { return name.replace(/\.[^.]+$/, "").replace(/[^\p{L}\p{N}_.-]+/gu, "_"); }
function currentPage() { return pages[currentPageIndex] || null; }
function hasLexicalContent(text) {
  return /[\p{L}\p{N}]/u.test(String(text || ""));
}

function transcriptionTokenInfo(page = currentPage()) {
  const text = (page?.sourceText || "").trim();
  if (!text) return {tokens: [], ignored: []};

  const raw = text.split(/\s+/u).filter(Boolean);
  const tokens = [];
  const ignored = [];
  let pendingOpening = "";

  const openingChars = new Set(["(", "[", "{", "«", "“", "\"", "׳", "״", "'"]);
  const closingChars = new Set([".", ",", ";", ":", "!", "?", "…", "%", ")", "]", "}", "»", "”", "\"", "׳", "״", "'"]);
  const dashChars = new Set(["-", "–", "—", ">", "<", "→", "←"]);
  const onlyOpening = t => [...t].length > 0 && [...t].every(ch => openingChars.has(ch));
  const onlyClosing = t => [...t].length > 0 && [...t].every(ch => closingChars.has(ch));
  const onlyDashLike = t => [...t].length > 0 && [...t].every(ch => dashChars.has(ch));

  for (const rawToken of raw) {
    let token = rawToken;
    if (hasLexicalContent(token)) {
      if (pendingOpening) { token = pendingOpening + token; pendingOpening = ""; }
      tokens.push(token);
      continue;
    }

    // Standalone punctuation is NOT counted as an extra word.
    // Opening punctuation is attached to the following lexical word.
    if (onlyOpening(token)) {
      pendingOpening += token;
      continue;
    }

    // Closing punctuation / separators are attached to the preceding word when possible.
    // This makes "word ." equivalent to "word." for segmentation.
    if (onlyClosing(token) || onlyDashLike(token)) {
      if (tokens.length) tokens[tokens.length - 1] += token;
      else ignored.push(token);
      continue;
    }

    // Unknown punctuation-only material must never shift all labels. Keep it only
    // if it can safely be attached to an existing word; otherwise ignore it.
    if (tokens.length) tokens[tokens.length - 1] += token;
    else ignored.push(token);
  }

  if (pendingOpening) {
    if (tokens.length) tokens[tokens.length - 1] += pendingOpening;
    else ignored.push(pendingOpening);
  }
  return {tokens, ignored};
}

function sourceWords(page = currentPage()) {
  return transcriptionTokenInfo(page).tokens;
}


function autoSettings() {
  const valueOr = (el, fallback) => {
    const v = Number(el?.value);
    return Number.isFinite(v) ? v : fallback;
  };
  return {
    inkThreshold: Math.max(40, Math.min(240, valueOr(inkThresholdInput, 145))),
    gapInkMax: Math.max(0, Math.min(20, valueOr(gapInkMaxInput, 1))),
    minGap: Math.max(1, Math.min(80, valueOr(minGapInput, 5))),
    minWordWidth: Math.max(5, Math.min(200, valueOr(minWordWidthInput, 14))),
    innerPad: 2,
  };
}

function getInkAnalysis(page) {
  const w = page.image.width, h = page.image.height;
  const off = document.createElement("canvas");
  off.width = w; off.height = h;
  const octx = off.getContext("2d", {willReadFrequently:true});
  octx.drawImage(page.image, 0, 0);
  const rgba = octx.getImageData(0, 0, w, h).data;
  const {inkThreshold} = autoSettings();
  const profile = new Uint16Array(w);
  const mask = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const i = (row + x) * 4;
      // Integer approximation of perceptual luminance.
      const lum = (77 * rgba[i] + 150 * rgba[i+1] + 29 * rgba[i+2]) >> 8;
      if (lum < inkThreshold) {
        mask[row + x] = 1;
        profile[x]++;
      }
    }
  }
  return {w, h, profile, mask};
}

function findLowInkRuns(profile, left, right, maxInk, minGap) {
  const runs = [];
  let x = left;
  while (x <= right) {
    if (profile[x] <= maxInk) {
      let end = x;
      let sum = profile[x];
      while (end + 1 <= right && profile[end + 1] <= maxInk) {
        end++;
        sum += profile[end];
      }
      const width = end - x + 1;
      if (width >= minGap && x > left && end < right) {
        runs.push({start:x, end, width, meanInk:sum / width, cut:Math.floor((x + end) / 2)});
      }
      x = end + 1;
    } else {
      x++;
    }
  }
  return runs;
}

function segmentLooksLikeWord(profile, a, b, minWordWidth) {
  if (b - a < minWordWidth) return false;
  let inkMass = 0, inkCols = 0, peak = 0;
  for (let x = a; x < b; x++) {
    const v = profile[x] || 0;
    inkMass += v;
    if (v > 0) inkCols++;
    if (v > peak) peak = v;
  }
  // Reject segments that are almost certainly just an isolated dot/comma/dash.
  // Keep the thresholds deliberately weak so short real Hebrew words are retained.
  if (inkMass < 18) return false;
  if (inkCols < 4 && peak < 7) return false;
  if (peak <= 3 && inkMass < 45) return false;
  return true;
}

function validCuts(profile, cuts, left, rightExclusive, minWordWidth) {
  const sorted = [...cuts].sort((a,b)=>a-b);
  const bounds = [left, ...sorted, rightExclusive];
  for (let i = 0; i < bounds.length - 1; i++) {
    if (!segmentLooksLikeWord(profile, bounds[i], bounds[i+1], minWordWidth)) return false;
  }
  return true;
}

function chooseCuts(profile, left, right, requestedWords, runs, minWordWidth) {
  if (requestedWords !== null && requestedWords <= 1) return {cuts:[], fallback:0};
  if (requestedWords === null) {
    return {cuts:runs.map(r=>r.cut).sort((a,b)=>a-b), fallback:0};
  }

  const target = requestedWords - 1;
  const chosen = [];
  // First prefer the widest genuinely empty/near-empty valleys.
  const ranked = [...runs].sort((a,b) => (b.width - a.width) || (a.meanInk - b.meanInk));
  for (const run of ranked) {
    if (validCuts(profile, [...chosen, run.cut], left, right + 1, minWordWidth)) {
      chosen.push(run.cut);
      if (chosen.length === target) break;
    }
  }

  // If notebook lines/noise broke a real gap into smaller pieces, fall back to
  // the lowest local ink-density columns while enforcing minimum word width.
  let fallback = 0;
  if (chosen.length < target) {
    const valleys = [];
    for (let x = left + minWordWidth; x <= right - minWordWidth; x++) {
      let sum = 0, count = 0;
      for (let dx = -2; dx <= 2; dx++) {
        const xx = x + dx;
        if (xx >= left && xx <= right) { sum += profile[xx]; count++; }
      }
      valleys.push({x, score:sum / Math.max(1,count)});
    }
    valleys.sort((a,b)=>a.score-b.score);
    for (const v of valleys) {
      if (chosen.some(c => Math.abs(c - v.x) < minWordWidth)) continue;
      if (!validCuts(profile, [...chosen, v.x], left, right + 1, minWordWidth)) continue;
      chosen.push(v.x);
      fallback++;
      if (chosen.length === target) break;
    }
  }
  return {cuts:chosen.sort((a,b)=>a-b), fallback};
}

function makeAutomaticBoxes(page) {
  const words = sourceWords(page);
  const requestedWords = words.length ? words.length : null;
  const settings = autoSettings();
  const {w, h, profile, mask} = getInkAnalysis(page);

  let left = -1, right = -1;
  for (let x = 0; x < w; x++) {
    if (profile[x] > 0) { if (left < 0) left = x; right = x; }
  }
  if (left < 0) return {boxes:[], requestedWords, fallback:0, gapRuns:0, error:"No dark ink was detected."};

  left = Math.max(0, left - settings.innerPad);
  right = Math.min(w - 1, right + settings.innerPad);
  const runs = findLowInkRuns(profile, left, right, settings.gapInkMax, settings.minGap);
  const {cuts, fallback} = chooseCuts(profile, left, right, requestedWords, runs, settings.minWordWidth);
  const bounds = [left, ...cuts, right + 1];
  const boxes = [];

  for (let i = 0; i < bounds.length - 1; i++) {
    const a = bounds[i], b = bounds[i+1];
    let firstInk = -1, lastInk = -1;
    for (let x = a; x < b; x++) {
      if (profile[x] > 0) {
        if (firstInk < 0) firstInk = x;
        lastInk = x;
      }
    }
    if (firstInk < 0) continue;
    let x0 = Math.max(0, firstInk - settings.innerPad);
    let x1 = Math.min(w, lastInk + 1 + settings.innerPad);
    if (x1 - x0 < MIN_BOX_WIDTH) {
      const need = MIN_BOX_WIDTH - (x1 - x0);
      x0 = Math.max(0, x0 - Math.ceil(need / 2));
      x1 = Math.min(w, x1 + Math.floor(need / 2));
      if (x1 - x0 < MIN_BOX_WIDTH) {
        if (x0 === 0) x1 = Math.min(w, MIN_BOX_WIDTH);
        else if (x1 === w) x0 = Math.max(0, w - MIN_BOX_WIDTH);
      }
    }
    boxes.push({
      id: crypto.randomUUID(),
      order: boxes.length + 1,
      x: x0,
      y: 0,
      width: x1 - x0,
      height: h,
      text: "",
      automatic: true,
    });
  }
  return {boxes, requestedWords, fallback, gapRuns:runs.length, error:null};
}

function assignWordsToPage(page) {
  const words = sourceWords(page);
  renumberBoxesByPosition(page);
  if (!words.length) return 0;
  const count = Math.min(words.length, page.boxes.length);
  for (let i = 0; i < count; i++) page.boxes[i].text = words[i];
  for (let i = count; i < page.boxes.length; i++) page.boxes[i].text = "";
  return count;
}

function autoSegmentPage(page, silent=false) {
  if (!page) return {ok:false, message:"No image loaded."};
  const result = makeAutomaticBoxes(page);
  if (result.error) {
    if (!silent) setStatus(result.error);
    return {ok:false, message:result.error};
  }
  page.boxes = result.boxes;
  const assigned = assignWordsToPage(page);
  page.autoInfo = result;
  const expected = result.requestedWords;
  const exact = expected === null || expected === page.boxes.length;
  const msg = expected === null
    ? `Auto-cut ${page.boxes.length} word candidate(s) from low-ink gaps.`
    : `Auto-cut ${page.boxes.length}/${expected} words${result.fallback ? ` (${result.fallback} fallback valley cut${result.fallback===1?"":"s"})` : ""}. Labels assigned: ${assigned}.`;
  if (!silent) {
    selectedBoxId = null;
    transcription.value = "";
    render();
    autoInfo.textContent = msg + (exact ? "" : " Check this line manually.");
    setStatus(msg);
  }
  return {ok:exact, message:msg, ...result};
}

async function autoSegmentAll(options={}) {
  if (!pages.length) return setStatus("No images loaded.");
  let segmented = 0, exact = 0, skipped = 0, fallbackCuts = 0;
  const onlyWithText = Boolean(options.onlyWithText);
  autoAllBtn.disabled = true;
  autoCurrentBtn.disabled = true;
  try {
    for (let i = 0; i < pages.length; i++) {
      const page = pages[i];
      if (onlyWithText && !sourceWords(page).length) { skipped++; continue; }
      const r = autoSegmentPage(page, true);
      if (r.boxes?.length) segmented++;
      if (r.ok) exact++;
      fallbackCuts += r.fallback || 0;
      if (i % 25 === 0) {
        setStatus(`Auto-cutting ${i+1}/${pages.length}...`);
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
    selectedBoxId = null;
    transcription.value = "";
    render();
    const msg = `Auto-cut finished: ${segmented} line(s), ${exact} exact word counts, ${fallbackCuts} fallback valley cut(s)${skipped ? `, ${skipped} skipped without TXT` : ""}.`;
    autoInfo.textContent = msg;
    setStatus(msg);
  } finally {
    autoAllBtn.disabled = false;
    autoCurrentBtn.disabled = false;
  }
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
  const bgWidth = Math.max(8, Math.min(box.width - 4, metrics.width + 8));
  const bgHeight = fontSize + 6;
  ctx.fillStyle = isSelected ? "rgba(15,118,110,.88)" : "rgba(255,255,255,.86)";
  ctx.fillRect(Math.max(box.x + 2, x - bgWidth), y - 2, bgWidth, bgHeight);
  ctx.fillStyle = isSelected ? "white" : "#111827";
  ctx.fillText(text, x, y);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

function boxHandles(box) {
  const s = HANDLE_SIZE;
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  const x2 = box.x + box.width, y2 = box.y + box.height;
  return [
    {name:"nw",x:box.x,y:box.y,cursor:"nwse-resize"},{name:"n",x:cx,y:box.y,cursor:"ns-resize"},
    {name:"ne",x:x2,y:box.y,cursor:"nesw-resize"},{name:"e",x:x2,y:cy,cursor:"ew-resize"},
    {name:"se",x:x2,y:y2,cursor:"nwse-resize"},{name:"s",x:cx,y:y2,cursor:"ns-resize"},
    {name:"sw",x:box.x,y:y2,cursor:"nesw-resize"},{name:"w",x:box.x,y:cy,cursor:"ew-resize"},
  ].map(h => ({...h,left:h.x-s/2,top:h.y-s/2,size:s}));
}
function drawResizeHandles(box) {
  ctx.save();
  for (const h of boxHandles(box)) {
    ctx.fillStyle="#fff"; ctx.strokeStyle="#0f766e"; ctx.lineWidth=2;
    ctx.fillRect(h.left,h.top,h.size,h.size); ctx.strokeRect(h.left,h.top,h.size,h.size);
  }
  ctx.restore();
}
function handleAtPoint(box,p) {
  if (!box) return null;
  return boxHandles(box).find(h => p.x>=h.left && p.x<=h.left+h.size && p.y>=h.top && p.y<=h.top+h.size) || null;
}
function selectedBox(page) { return page?.boxes.find(b => b.id === selectedBoxId) || null; }
function resizedRect(startBox, handleName, p) {
  let x1=startBox.x,y1=startBox.y,x2=startBox.x+startBox.width,y2=startBox.y+startBox.height;
  if (handleName.includes("w")) x1=p.x; if (handleName.includes("e")) x2=p.x;
  if (handleName.includes("n")) y1=p.y; if (handleName.includes("s")) y2=p.y;
  if (x2-x1<MIN_BOX_WIDTH) { if(handleName.includes("w")) x1=x2-MIN_BOX_WIDTH; else x2=x1+MIN_BOX_WIDTH; }
  if (y2-y1<MIN_BOX_HEIGHT) { if(handleName.includes("n")) y1=y2-MIN_BOX_HEIGHT; else y2=y1+MIN_BOX_HEIGHT; }
  x1=Math.max(0,Math.min(canvas.width-MIN_BOX_WIDTH,x1)); y1=Math.max(0,Math.min(canvas.height-MIN_BOX_HEIGHT,y1));
  x2=Math.max(x1+MIN_BOX_WIDTH,Math.min(canvas.width,x2)); y2=Math.max(y1+MIN_BOX_HEIGHT,Math.min(canvas.height,y2));
  return {x:Math.round(x1),y:Math.round(y1),width:Math.round(x2-x1),height:Math.round(y2-y1)};
}

function sortedBoxesByPosition(boxes) {
  return [...boxes].sort((a,b) => {
    const ay=a.y+a.height/2, by=b.y+b.height/2;
    const tol=Math.max(10,Math.min(a.height,b.height)*0.6);
    if (Math.abs(ay-by)>tol) return ay-by;
    const ax=a.x+a.width/2, bx=b.x+b.width/2;
    return bx-ax; // Hebrew/RTL: rightmost word first.
  });
}
function renumberBoxesByPosition(page) {
  if(!page) return [];
  page.boxes=sortedBoxesByPosition(page.boxes);
  page.boxes.forEach((b,i)=>b.order=i+1);
  return page.boxes;
}

function redraw() {
  const page=currentPage();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if(!page) return;
  ctx.drawImage(page.image,0,0);
  renumberBoxesByPosition(page);
  for(const box of page.boxes){
    const selected=selectedBoxId===box.id;
    ctx.lineWidth=selected?4:2; ctx.strokeStyle=selected?"#0f766e":"#dc2626";
    ctx.strokeRect(box.x,box.y,box.width,box.height);
    ctx.fillStyle=selected?"#0f766e":"#dc2626"; ctx.font="16px Arial";
    ctx.fillText(String(box.order),box.x+3,Math.max(17,box.y-5));
    drawBoxText(box,selected);
    if(selected) drawResizeHandles(box);
  }
  if(drawing){ ctx.lineWidth=2; ctx.strokeStyle="#2563eb"; ctx.setLineDash([8,5]); ctx.strokeRect(drawing.x,drawing.y,drawing.width,drawing.height); ctx.setLineDash([]); }
}

function renderBoxList() {
  const page=currentPage(); boxList.innerHTML=""; if(!page) return;
  renumberBoxesByPosition(page);
  for(const box of page.boxes){
    const li=document.createElement("li");
    li.textContent=`${box.order}. ${box.text || "(no word)"}`;
    if(box.id===selectedBoxId) li.classList.add("selected");
    li.addEventListener("click",()=>selectBox(box.id)); boxList.appendChild(li);
  }
}
function updatePageSelect(){
  pageSelect.innerHTML="";
  pages.forEach((p,i)=>{ const o=document.createElement("option"); o.value=String(i); o.textContent=`${i+1}. ${p.file.name}`; pageSelect.appendChild(o); });
  pageSelect.value=String(currentPageIndex);
}
function updateWordCount(){
  const page=currentPage();
  if(!page){ wordCountInfo.textContent="Load a TXT pair and use Auto-cut, or draw boxes manually."; assignWordsBtn.disabled=true; return; }
  const info=transcriptionTokenInfo(page);
  const words=info.tokens;
  assignWordsBtn.disabled=!words.length || !page.boxes.length;
  if(!words.length) wordCountInfo.textContent=`Boxes: ${page.boxes.length}. No usable word tokens in TXT; punctuation-only material will not be saved.`;
  else {
    const match=words.length===page.boxes.length;
    const ignored=info.ignored.length ? ` | ignored standalone punctuation: ${info.ignored.join(" ")}` : "";
    wordCountInfo.textContent=`Usable TXT words: ${words.length} | boxes: ${page.boxes.length}${ignored}${match ? " — counts match ✓" : ""}`;
    wordCountInfo.classList.toggle("ok",match);
  }
}
function render(){
  const page=currentPage(), hasPage=Boolean(page);
  pageInfo.textContent=hasPage?`${page.image.width} x ${page.image.height}, word boxes: ${page.boxes.length}`:"No image loaded.";
  sourceTranscription.value=page?.sourceText || "";
  assignWordsBtn.disabled=!hasPage || !sourceWords(page).length || !page.boxes.length;
  transcription.disabled=!hasPage || !selectedBoxId;
  deleteBtn.disabled=!hasPage || !selectedBoxId;
  savePageBtn.disabled=!hasPage; saveNextBtn.disabled=!hasPage; saveAllBtn.disabled=!pages.length;
  prevBtn.disabled=currentPageIndex<=0; nextBtn.disabled=currentPageIndex<0 || currentPageIndex>=pages.length-1;
  if(page){ canvas.width=page.image.width; canvas.height=page.image.height; } else { canvas.width=1; canvas.height=1; }
  canvas.style.transform=`scale(${Number(zoomInput.value)/100})`; canvas.style.transformOrigin="top left";
  renderBoxList(); updateWordCount(); redraw();
}

function scrollBoxIntoView(box,force=false){
  if(!box||!canvasWrap) return; const scale=Number(zoomInput.value)/100, margin=40;
  const left=box.x*scale,top=box.y*scale,right=(box.x+box.width)*scale,bottom=(box.y+box.height)*scale;
  if(force){canvasWrap.scrollLeft=Math.max(0,left-margin);canvasWrap.scrollTop=Math.max(0,top-margin);canvas.scrollIntoView({block:"nearest",inline:"nearest"});return;}
  const vl=canvasWrap.scrollLeft,vt=canvasWrap.scrollTop,vr=vl+canvasWrap.clientWidth,vb=vt+canvasWrap.clientHeight;
  if(left<vl+margin)canvasWrap.scrollLeft=Math.max(0,left-margin); else if(right>vr-margin)canvasWrap.scrollLeft=Math.max(0,right-canvasWrap.clientWidth+margin);
  if(top<vt+margin)canvasWrap.scrollTop=Math.max(0,top-margin); else if(bottom>vb-margin)canvasWrap.scrollTop=Math.max(0,bottom-canvasWrap.clientHeight+margin);
}
function focusTranscription(selectText=true){ setTimeout(()=>{transcription.focus({preventScroll:true}); if(selectText) transcription.select();},0); }
function selectBox(id){
  const page=currentPage(); const box=page?.boxes.find(b=>b.id===id);
  selectedBoxId=box?id:null; transcription.disabled=!box; transcription.value=box?box.text:""; renderBoxList(); redraw();
  if(box){focusTranscription(false);setTimeout(()=>scrollBoxIntoView(box),0);}
}

function isImageFile(f){return f.type.startsWith("image/")||/\.(png|jpe?g|webp|bmp)$/i.test(f.name);}
function isPdfFile(f){return f.type==="application/pdf"||/\.pdf$/i.test(f.name);}
function isTextFile(f){return f.type.startsWith("text/")||/\.txt$/i.test(f.name);}
async function readTextFile(file){return await file.text();}
async function loadImageFile(file, sourceText=""){
  const url=URL.createObjectURL(file);
  try{
    const image=new Image(); await new Promise((res,rej)=>{image.onload=res;image.onerror=rej;image.src=url;});
    return {file,image,boxes:[],baseName:safeBaseName(file.name),sourceText:sourceText.replace(/\r?\n/g," ").trim()};
  }finally{URL.revokeObjectURL(url);}
}
function fileToDataUrl(file){return new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=()=>rej(r.error||new Error("Could not read file"));r.readAsDataURL(file);});}
async function renderPdfFile(file){
  setStatus(`Rendering PDF: ${file.name}...`);
  const response=await fetch("/render_pdf",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:file.name,data_url:await fileToDataUrl(file),scale:2.0})});
  const result=await response.json().catch(()=>({})); if(!response.ok||!result.ok)throw new Error(result.error||`HTTP ${response.status}`);
  const out=[]; for(let i=0;i<result.pages.length;i++){ const item=result.pages[i]; const blob=await(await fetch(item.data_url)).blob(); const imageFile=new File([blob],item.name,{type:"image/png"}); out.push(await loadImageFile(imageFile)); }
  return out;
}
async function loadImages(files){
  pages=[]; selectedBoxId=null;
  for(const f of files.filter(isImageFile)) pages.push(await loadImageFile(f));
  currentPageIndex=pages.length?0:-1; updatePageSelect(); render(); setStatus(pages.length?`Loaded ${pages.length} image(s).`:"No supported images loaded.");
}
async function loadDatasetFolder(files){
  pages=[]; selectedBoxId=null;
  const all=[...files]; const textByStem=new Map();
  for(const f of all.filter(isTextFile)){
    const rel=f.webkitRelativePath||f.name; const stem=rel.replace(/\.[^.]+$/,""); textByStem.set(stem,await readTextFile(f));
  }
  const images=all.filter(isImageFile).sort((a,b)=>(a.webkitRelativePath||a.name).localeCompare(b.webkitRelativePath||b.name,undefined,{numeric:true}));
  for(const f of images){ const rel=f.webkitRelativePath||f.name; const stem=rel.replace(/\.[^.]+$/,""); pages.push(await loadImageFile(f,textByStem.get(stem)||"")); }
  currentPageIndex=pages.length?0:-1; updatePageSelect(); render();
  const paired=pages.filter(p=>p.sourceText).length;
  setStatus(`Loaded ${pages.length} image(s); ${paired} have matching TXT transcription.`);
  if (pages.length && paired && autoAfterLoad?.checked) {
    await autoSegmentAll({onlyWithText:true});
  }
}
async function loadPdfs(files){
  pages=[]; selectedBoxId=null;
  for(const f of files.filter(isPdfFile)) pages.push(...await renderPdfFile(f));
  currentPageIndex=pages.length?0:-1; updatePageSelect(); render(); setStatus(`Loaded ${pages.length} rendered PDF page(s).`);
}

chooseDatasetFolderBtn.addEventListener("click",()=>{datasetFolderInput.value="";datasetFolderInput.click();});
chooseImagesBtn.addEventListener("click",()=>{fileInput.value="";fileInput.click();});
choosePdfBtn.addEventListener("click",()=>{pdfInput.value="";pdfInput.click();});
datasetFolderInput.addEventListener("change",async()=>{try{await loadDatasetFolder(datasetFolderInput.files);}catch(e){console.error(e);setStatus(`Load failed: ${e.message}`);}});
fileInput.addEventListener("change",async()=>{try{await loadImages([...fileInput.files]);}catch(e){console.error(e);setStatus(`Load failed: ${e.message}`);}});
pdfInput.addEventListener("change",async()=>{try{await loadPdfs([...pdfInput.files]);}catch(e){console.error(e);setStatus(`Load failed: ${e.message}`);}});

function changePage(index){ currentPageIndex=index; selectedBoxId=null; transcription.value=""; updatePageSelect(); render(); }
pageSelect.addEventListener("change",()=>changePage(Number(pageSelect.value)));
prevBtn.addEventListener("click",()=>{if(currentPageIndex>0)changePage(currentPageIndex-1);});
nextBtn.addEventListener("click",()=>{if(currentPageIndex<pages.length-1)changePage(currentPageIndex+1);});
zoomInput.addEventListener("input",render);

sourceTranscription.addEventListener("input",()=>{ const page=currentPage(); if(!page)return; page.sourceText=sourceTranscription.value.replace(/\r?\n/g," ").trim(); updateWordCount(); assignWordsBtn.disabled=!sourceWords(page).length||!page.boxes.length; });
assignWordsBtn.addEventListener("click",()=>{
  const page=currentPage(); if(!page)return; const words=sourceWords(page); renumberBoxesByPosition(page);
  if(!words.length)return setStatus("No line transcription is available.");
  const count=Math.min(words.length,page.boxes.length);
  for(let i=0;i<count;i++) page.boxes[i].text=words[i];
  if(page.boxes.length>words.length) for(let i=words.length;i<page.boxes.length;i++) page.boxes[i].text="";
  render();
  if(words.length===page.boxes.length) setStatus(`Assigned all ${words.length} words from right to left.`);
  else setStatus(`Assigned ${count} labels, but TXT has ${words.length} words and there are ${page.boxes.length} boxes. Check segmentation.`);
});

autoCurrentBtn.addEventListener("click",()=>autoSegmentPage(currentPage()));
autoAllBtn.addEventListener("click",()=>autoSegmentAll({onlyWithText:false}));
clearBoxesBtn.addEventListener("click",()=>{
  const page=currentPage();
  if(!page)return;
  page.boxes=[]; page.autoInfo=null; selectedBoxId=null; transcription.value="";
  render();
  autoInfo.textContent="Boxes cleared. You can auto-cut again or draw manually.";
  setStatus("Cleared word boxes on current line.");
});

function canvasPoint(event){const r=canvas.getBoundingClientRect();return{x:Math.round((event.clientX-r.left)*(canvas.width/r.width)),y:Math.round((event.clientY-r.top)*(canvas.height/r.height))};}
function boxAtPoint(page,p){if(!page)return null;for(let i=page.boxes.length-1;i>=0;i--){const b=page.boxes[i];if(p.x>=b.x&&p.x<=b.x+b.width&&p.y>=b.y&&p.y<=b.y+b.height)return b;}return null;}
canvas.addEventListener("contextmenu",e=>e.preventDefault());
canvas.addEventListener("mousedown",event=>{
  const page=currentPage(); if(!page)return; const p=canvasPoint(event), active=selectedBox(page), h=handleAtPoint(active,p), clicked=boxAtPoint(page,p);
  if(event.button===0){
    if(active&&h){resizing={boxId:active.id,handle:h.name,startBox:{...active}};canvas.style.cursor=h.cursor;return;}
    if(clicked){selectBox(clicked.id);return;}
    drawing={startX:p.x,startY:p.y,x:p.x,y:p.y,width:0,height:0};
  } else if(event.button===2){if(clicked)selectedBoxId=clicked.id;deleteSelectedBox();}
});
canvas.addEventListener("mousemove",event=>{
  const page=currentPage();if(!page)return;const p=canvasPoint(event);
  if(resizing){const b=page.boxes.find(x=>x.id===resizing.boxId);if(!b)return;Object.assign(b,resizedRect(resizing.startBox,resizing.handle,p));renderBoxList();updateWordCount();redraw();return;}
  if(drawing){drawing.x=Math.min(drawing.startX,p.x);drawing.y=Math.min(drawing.startY,p.y);drawing.width=Math.abs(p.x-drawing.startX);drawing.height=Math.abs(p.y-drawing.startY);redraw();return;}
  const h=handleAtPoint(selectedBox(page),p);canvas.style.cursor=h?h.cursor:"crosshair";
});
function finishResize(){const page=currentPage();if(!resizing)return false;const b=page?.boxes.find(x=>x.id===resizing.boxId);resizing=null;canvas.style.cursor="crosshair";if(b)setStatus(`Resized word box ${b.order}.`);redraw();return true;}
canvas.addEventListener("mouseleave",()=>{if(drawing){drawing=null;redraw();}finishResize();});
canvas.addEventListener("mouseup",()=>{
  const page=currentPage();if(finishResize())return;if(!page||!drawing)return;const r=drawing;drawing=null;
  if(r.width<MIN_BOX_WIDTH||r.height<MIN_BOX_HEIGHT){redraw();return setStatus(`Box too small. Minimum ${MIN_BOX_WIDTH} x ${MIN_BOX_HEIGHT}px.`);}
  const box={id:crypto.randomUUID(),order:page.boxes.length+1,x:r.x,y:r.y,width:r.width,height:r.height,text:""};page.boxes.push(box);renumberBoxesByPosition(page);selectBox(box.id);updateWordCount();
  const words=sourceWords(page); setStatus(words.length===page.boxes.length?"Word counts now match. Click 'Assign words from line TXT'.":"Added word box.");
});

transcription.addEventListener("input",()=>{const page=currentPage(),box=page?.boxes.find(b=>b.id===selectedBoxId);if(!box)return;box.text=transcription.value.trim();renderBoxList();redraw();});
transcription.addEventListener("keydown",event=>{if(event.key!=="Enter"||event.shiftKey)return;event.preventDefault();selectNextBox();});
function selectNextBox(){const page=currentPage();if(!page||!page.boxes.length)return false;renumberBoxesByPosition(page);const i=page.boxes.findIndex(b=>b.id===selectedBoxId);const next=page.boxes[Math.min(i+1,page.boxes.length-1)];if(!next||next.id===selectedBoxId)return false;selectBox(next.id);setTimeout(()=>scrollBoxIntoView(next,true),0);return true;}
function deleteSelectedBox(){const page=currentPage();if(!page||!selectedBoxId)return false;const b=page.boxes.find(x=>x.id===selectedBoxId);page.boxes=page.boxes.filter(x=>x.id!==selectedBoxId);selectedBoxId=null;transcription.value="";render();setStatus(b?`Deleted word box ${b.order}.`:"Deleted word box.");return true;}
document.addEventListener("keydown",event=>{if(event.key!=="Tab")return;if(deleteSelectedBox())event.preventDefault();});
deleteBtn.addEventListener("click",deleteSelectedBox);
toggleBoxTextBtn.addEventListener("click",()=>{showBoxText=!showBoxText;toggleBoxTextBtn.textContent=showBoxText?"Hide text in boxes":"Show text in boxes";redraw();});
helpBtn.addEventListener("click",()=>{helpPanel.hidden=!helpPanel.hidden;});

function cropToBlob(page,box){
  const pad=Math.max(0,Math.min(50,Number(paddingInput.value)||0));
  const x=Math.max(0,Math.floor(box.x-pad)), y=Math.max(0,Math.floor(box.y-pad));
  const x2=Math.min(page.image.width,Math.ceil(box.x+box.width+pad)), y2=Math.min(page.image.height,Math.ceil(box.y+box.height+pad));
  const w=Math.max(1,x2-x),h=Math.max(1,y2-y); const crop=document.createElement("canvas");crop.width=w;crop.height=h;
  crop.getContext("2d").drawImage(page.image,x,y,w,h,0,0,w,h); return new Promise(res=>crop.toBlob(res,"image/png"));
}
function blobToDataUrl(blob){return new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result);r.onerror=rej;r.readAsDataURL(blob);});}
async function saveFilesToServer(folder,files){const response=await fetch("/save_page",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({output_folder:folder,files})});const result=await response.json().catch(()=>({}));if(!response.ok||!result.ok)throw new Error(result.error||`HTTP ${response.status}`);return result;}
function validatePageForSave(page) {
  if (!page) return {ok:false, reason:"No image loaded."};
  if (!page.boxes.length) return {ok:false, reason:"No word boxes on this image."};
  renumberBoxesByPosition(page);
  const expected = sourceWords(page);
  const unlabeled = page.boxes.filter(b => !b.text.trim());
  if (unlabeled.length) return {ok:false, reason:`${unlabeled.length} box(es) have no label.`};

  const usableBoxes = page.boxes.filter(b => hasLexicalContent(b.text));
  const punctuationOnly = page.boxes.filter(b => b.text.trim() && !hasLexicalContent(b.text));

  // With a paired TXT, never save a line if the lexical word count is inconsistent.
  // It is safer to skip one questionable line than to create shifted/corrupted labels.
  if (expected.length && usableBoxes.length !== expected.length) {
    return {ok:false, reason:`Unsafe count mismatch: ${usableBoxes.length} usable boxes vs ${expected.length} TXT words.`, punctuationOnly};
  }
  if (!usableBoxes.length) return {ok:false, reason:"No lexical word crops to save.", punctuationOnly};
  return {ok:true, usableBoxes, punctuationOnly};
}

async function savePage(page, options={}){
  const out=outputPath.value.trim();
  if(!out) { if(options.skipUnsafe) return {skipped:true, reason:"Output folder is empty."}; throw new Error("Please type an output folder."); }
  const check=validatePageForSave(page);
  if(!check.ok) {
    if(options.skipUnsafe) return {skipped:true, reason:check.reason, skippedPunctuation:check.punctuationOnly?.length||0};
    throw new Error(check.reason);
  }
  const files=[];
  let savedIndex=0;
  for(const box of check.usableBoxes){
    savedIndex++;
    const stem=`${page.baseName}_word_${String(savedIndex).padStart(4,"0")}`;
    const blob=await cropToBlob(page,box);
    files.push({name:`${stem}.png`,kind:"base64",data_url:await blobToDataUrl(blob)});
    files.push({name:`${stem}.txt`,kind:"text",text:box.text.trim()+"\n"});
  }
  const result=await saveFilesToServer(out,files);
  const punctMsg=check.punctuationOnly.length ? ` Skipped ${check.punctuationOnly.length} punctuation-only crop(s).` : "";
  if(!options.silent) setStatus(`Saved ${savedIndex} clean word crop(s) to ${result.output_folder}.${punctMsg}`);
  return {...result, savedCount:savedIndex, skippedPunctuation:check.punctuationOnly.length};
}
savePageBtn.addEventListener("click",async()=>{try{await savePage(currentPage());}catch(e){console.error(e);setStatus(`Save blocked for safety: ${e.message}`);}});
saveNextBtn.addEventListener("click",async()=>{try{await savePage(currentPage());if(currentPageIndex<pages.length-1)changePage(currentPageIndex+1);}catch(e){console.error(e);setStatus(`Save blocked for safety: ${e.message}`);}});
saveAllBtn.addEventListener("click",async()=>{
  try{
    let total=0, skippedLines=0, skippedPunctuation=0; const reasons=[];
    for(const page of pages){
      if(!page.boxes.length){skippedLines++;continue;}
      const r=await savePage(page,{skipUnsafe:true,silent:true});
      if(r?.skipped){skippedLines++; if(reasons.length<5) reasons.push(`${page.file.name}: ${r.reason}`); continue;}
      total+=r.savedCount||0; skippedPunctuation+=r.skippedPunctuation||0;
    }
    const detail=reasons.length ? ` First skipped: ${reasons.join(" | ")}` : "";
    setStatus(`Saved ${total} clean word crop(s). Skipped ${skippedLines} unsafe/empty line(s) and ${skippedPunctuation} punctuation-only crop(s).${detail}`);
  }catch(e){console.error(e);setStatus(`Save failed: ${e.message}`);}
});

render();
