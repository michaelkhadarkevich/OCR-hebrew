# Hebrew handwriting dataset segmentation

These two local browser tools support the dataset preparation workflow used in this project:

`PDF / page images -> manually labeled lines -> automatically proposed word crops -> reviewed PNG/TXT pairs`

| Tool | Source version | Start on Windows | Address |
|---|---|---|---|
| [Line segmenter](line_segmenter/) | Manual Segmenter PDF Portable v4 | `line_segmenter/Start_Manual_Segmenter.bat` | http://localhost:8770 |
| [Word segmenter](word_segmenter/) | Word Segmenter Auto v3 Punctuation Safe | `word_segmenter/Start_Word_Segmenter.bat` | http://localhost:8773 |

The folders contain unpacked source code from the original portable ZIPs. The older word segmenter v2 is superseded by v3. The repository's original `manual_segmenter/` remains available as the earlier image annotation tool.

## Requirements and startup

Install Python 3, then double-click the relevant start script. It opens the local application in your browser. If PDF support is missing, the script installs PyMuPDF using pip; this installation requires internet access. Document processing itself runs locally.

Alternatively, from the repository root:

```powershell
python -m pip install -r tools/segmentation/line_segmenter/requirements.txt
python tools/segmentation/line_segmenter/server.py
# In another terminal, when needed:
python tools/segmentation/word_segmenter/server.py
```

Both tools use the same Python dependency. Keep the server terminal open while working; close it or press Ctrl+C to stop. Each folder also includes the original Windows stop script.

## 1. Create the line dataset

1. Open the line segmenter and load a PDF, page image, or image folder.
2. Draw a box around each complete text line and enter its exact Hebrew transcription in logical reading order.
3. Preserve spaces between words in the transcription: the word segmenter needs them to determine the target word count.
4. Set an explicit output folder, for example an absolute path to `data/my_lines` in your checkout. Replace the UI's `YOUR_USER` example path.
5. Save and inspect the resulting same-name PNG/TXT pairs.

Keep crops in their natural reading orientation. The model's Hebrew data pipeline handles horizontal mirroring separately. Split source pages into train, validation and final-test groups before segmentation, and keep words from each line in the same split as that line.

## 2. Split lines into words

1. Open the word segmenter and choose the folder containing matching line images and UTF-8 TXT files.
2. Automatic cutting runs after loading by default. It counts dark pixels in image columns and proposes cuts in low-ink gaps, guided by the number of words in the transcription.
3. Review the proposed boxes and their labels in Hebrew right-to-left order. Resize, delete or replace boxes where necessary; adjust the ink threshold and cutting settings for the source handwriting.
4. Choose a separate output folder, for example an absolute path to `data/my_words`.
5. Use **Save words**, **Save + Next**, or **Save all loaded images**.

Output pairs are named like `source_word_0001.png` and `source_word_0001.txt`. Input line files are not modified when output is saved to a separate folder.

### Punctuation handling and review

Version 3 attaches standalone punctuation to neighboring lexical words where possible and excludes punctuation-only crops. It blocks/skips lines when the usable box count disagrees with the usable transcription word count.

Matching counts do not prove that every cut is correct. Touching words, faint strokes, notebook grids and unusual spacing can still require manual correction. No measured segmentation accuracy is claimed here; review crops and labels before training. Preserve the original spaced line transcriptions even though the recognition pipeline later removes whitespace.

## Source notes

- Line tool: `manual_segmenter_pdf_v4_portable.zip`.
- Word tool: `word_segmenter_auto_v3_punctuation_safe.zip`.
- Original tool-specific instructions are retained as `README.txt` in each folder.
- ZIP archives, Python caches, runtime PID files and generated datasets are excluded from this addition.
