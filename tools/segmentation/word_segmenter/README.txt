Word Segmenter - Auto v3 (Punctuation Safe)

Purpose
-------
Automatic word-level segmentation for handwritten-text line datasets, with extra safeguards for punctuation.
The app never changes the original input files.

Punctuation rules
-----------------
- A space before punctuation does NOT count as a word gap. Example: "מורפה ." is treated as two characters groups belonging to one word label: "מורפה.".
- Standalone closing punctuation such as . , : ; ! ? is attached to the preceding lexical word.
- Standalone opening quotes/brackets are attached to the following lexical word when possible.
- Standalone dash/separator punctuation is attached to a neighboring lexical word when safe.
- Punctuation-only material that cannot safely be assigned is ignored instead of shifting all word labels.
- At save time, punctuation-only boxes are never written to the word dataset.
- If the usable box count disagrees with the usable TXT word count, that line is blocked/skipped rather than saving corrupted labels.

Automatic cutting
-----------------
1. Pixels darker than the Ink Threshold are treated as handwriting ink.
2. For each X column, the app counts dark pixels (vertical projection).
3. Low-ink runs are candidate spaces.
4. The cleaned TXT transcription supplies the target number of real word tokens.
5. Candidate cut sets that create a tiny punctuation-like segment are rejected.
6. Boxes are ordered RIGHT-TO-LEFT for Hebrew.
7. Every box can still be resized/deleted/replaced manually.

Recommended workflow
--------------------
1. Extract this ZIP to a Windows folder.
2. Double-click Start_Word_Segmenter.bat.
3. The app opens at http://localhost:8773
4. Choose the dataset folder containing matching PNG/JPG + TXT pairs.
5. Auto-cut all lines is enabled by default.
6. Review any suspicious lines.
7. Enter an output folder.
8. Click Save all loaded images. Unsafe lines are skipped automatically.

Output
------
Each safe lexical word is saved as a cropped PNG + UTF-8 TXT:
    line_001_word_0001.png
    line_001_word_0001.txt

No external upload is used.
