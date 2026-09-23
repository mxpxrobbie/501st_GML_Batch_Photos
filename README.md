# 501st Legion GWL/GML Photo Processing Tool

An automated batch processing script designed for Garrison Membership Liaisons (GMLs) to prepare member photos for 501st Legion database submissions. 

It handles AI background removal, image scaling, composite framing, target file-size auto-tuning, editable 3-layer PSD output (for manual adjustment of output in Photoshop), GIF thumbnail creation, and ZIP packaging.

---

## Prerequisites

* **Python 3.10 or higher**
* **Windows OS** (if using the included `.bat` quick launcher)

---

## Directory Structure

Set up your base processing directory as follows (default drive/path: `Z:\501st`):

```text
Z:\501st\
├── 501st Framesets\
│   ├── TK_Stormtrooper\
│   │   ├── tk_full.psd
│   │   ├── tk_head.psd
│   │   └── tk_thumb.psd
│   ├── SL_DarthVader\
│   │   ├── sl_full.psd
│   │   ├── sl_head.psd
│   │   └── sl_thumb.psd
│   └── ...
├── TK8231\
│   ├── tk8231_full.jpg
│   ├── tk8231_head.jpg
│   └── tk8231_thumb.jpg
├── SL12345\
│   ├── photo1.jpg
│   └── photo2.png
├── process_501st_batch.py
└── Run_GWL_Batch.bat
```

### Folder Roles
* **`501st Framesets/`**: Contains subfolders for each costume prefix.
* **`[Member Folders]`**: Individual folders named by prefix + ID (e.g., `TK8231`, `SL12345`).
* **`complete/`**: Automatically created inside each member's folder during processing to store generated files.
* **`templates/`, `temp/`**: Automatically ignored by the script.

---

## File Naming Conventions

### 1. Framesets
Inside `501st Framesets/<Prefix_Folder>/`, the PSD frame files **must** end with:
* `_full.psd` — Full costume frame
* `_head.psd` — Head/bucket shot frame
* `_thumb.psd` — Thumbnail frame

### 2. Member Folders & Raw Photos
* **Folder Name**: Must begin with the costume prefix (e.g., `TK8231` -> `TK`, `ID50100` -> `ID`).
* **Raw Photos**: File naming keywords help auto-classify images:
  * **Full**: Contains `full` or `body`
  * **Head**: Contains `head`, `face`, `helmet`, or `bucket`
  * **Thumb**: Contains `thumb` or `small`
  * *Note: If no keywords are matched, photos fall back gracefully in directory order.*

---

## Installation & Setup

1. **Clone or download** this repository to your processing directory (e.g., `Z:\501st`).
2. Verify Python is installed and added to your system `PATH`.
3. Open `process_501st_batch.py` in a text editor to update `BASE_DIR` if your working folder is not `Z:\501st`:
   ```python
   BASE_DIR = Path(r"Z:\501st")
   ```

---

## Running the Processor

Double-click **`Run_GWL_Batch.bat`**. 

The batch file will automatically install or verify all required Python libraries on first run:
```bat
pillow rembg psd-tools opencv-python onnxruntime pytoshop numpy six packbits tqdm
```

### Output Workflow
For each processed folder, the script performs the following:
1. **AI Background Removal**: Removes original photo backgrounds via `rembg` (u2net model).
2. **Subject Auto-Inset**: Scales member photo to ~76% of canvas size to eliminate border clip-offs.
3. **Optimized Files**:
   * **Full**: Target size 20–50 KB (JPEG auto-tuned starting at 95% quality downward).
   * **Head**: Target size 10–40 KB (JPEG auto-tuned starting at 95% quality downward).
   * **Thumb**: Interlaced 64-color GIF thumbnail.
4. **Editable PSD Files**: Generates 3-layer working PSD files (`_working.psd`) structured top-to-bottom:
   * Layer 1 (Top): `Frame Overlay`
   * Layer 2 (Middle): `Trooper Photo`
   * Layer 3 (Bottom): `Background`
5. **ZIP Archive**: Packages `<member_id>_full.jpg`, `<member_id>_head.jpg`, and `<member_id>_thumb.gif` into `<member_id>.zip`.

---

## Generated Output Example (`TK8231/complete/`)

```text
TK8231/complete/
├── tk8231_full.jpg
├── tk8231_head.jpg
├── tk8231_thumb.gif
├── tk8231_full_working.psd
├── tk8231_head_working.psd
├── tk8231_thumb_working.psd
└── tk8231.zip
```
