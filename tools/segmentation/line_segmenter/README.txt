Manual Segmenter - PDF Portable v4

This is a standalone test build. It does NOT replace or delete any previous installation.

HOW TO RUN ON WINDOWS
1. Extract the ZIP to a normal folder.
2. Double-click Start_Manual_Segmenter.bat.
3. The app opens at http://localhost:8770
4. Confirm that the header says "PDF Portable v4".
5. Click "Choose PDF" and choose a PDF.

PDF support
- Each PDF page is rendered locally into an image.
- PyMuPDF is used locally on your computer.
- If PyMuPDF is missing, the Start script asks Python/pip to install it.
- The PDF is not uploaded to an external service.

You can still use images and image folders as before.
Output crops and TXT files are saved only to the output folder you enter.

This build intentionally uses port 8770 so it can run alongside the older build on 8765/8766.
