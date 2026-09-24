# 501st GML Batch Photo Processor

An automated Python tool designed for 501st Legion **Garrison Membership Liaisons (GMLs)** to streamline the processing, compositing, cropping, and web-optimization of member costume submission photos.

This tool automates background removal using AI, applies specialized cropping for full-body and helmet-off close-ups, aligns the member's face horizontally regardless of pose asymmetry, layers the subject inside official GML framesets, and generates both optimized web assets (JPG/GIF) and editable 3-layer Photoshop PSDs.

---

## Features

* **AI-Powered Background Removal**: Integrated with `rembg` and BiRefNet for automated, studio-grade background masking and shadow filtering.
* **Smart Framing & Proportional Fits**:
  * **Headshots (`HEAD`)**: Scaled vertically to fill **88% of the frame window height** so helmet-off portraits fill the view while letting shoulders extend cleanly underneath the frame overlay.
  * **Full Body / Thumbnails (`FULL` / `THUMB`)**: Scaled to 85% width and anchored nicely inside the lower frame border.
* **Head-Centered Horizontal Alignment**: Uses a specialized facial isolation algorithm (`get_head_center_x`) analyzing the top 35% of the cutout to ensure the trooper's head and face remain centered in the frame regardless of pose angles, shoulder armor, or prop width.
* **Editable Multi-Layer PSD Generator**: Outputs native 3-layer working PSDs (`Frame Overlay` -> `Trooper Photo` -> `Background`) using `pytoshop` with a custom `packbits` patch.
* **Automatic Web Optimization**:
  * Auto-tunes JPG compression to stay within target file size ranges (20–50 KB for full, 10–40 KB for head).
  * Auto-quantizes thumbnail images into interlaced 64-color GIFs (`_thumb.gif`).
* **Automated Packaging**: Creates a neat `.zip` archive per member containing all final submission images.

---

## Directory & File Requirements

The processor operates on a master base directory containing individual member folders and a shared `501st Framesets` directory.

### Directory Layout

```text
Z:\501st\
├── 501st Framesets\          <-- Master PSD Templates Folder
│   ├── TK\
│   │   ├── tk_full.psd
│   │   ├── tk_head.psd
│   │   └── tk_thumb.psd
│   ├── ID\
│   └── ...
├── TK8231\                   <-- Member Folder ([CostumePrefix][ID])
│   ├── full.jpg              <-- Full costume photo
│   ├── head.jpg              <-- Helmet-off photo
│   └── thumb.jpg (optional)  <-- Fallback photo
└── ...
```

### Folder & Naming Rules
1. **Member Folders**: Folder names must start with the costume prefix (e.g., `TK8231`, `ID11042`, `BH5501`).
2. **Framesets**: The script extracts the costume prefix (e.g., `TK`) and matches it to a subfolder inside `501st Framesets/`.
3. **Template PSDs**: Frameset folders must contain PSD files ending with `_full.psd`, `_head.psd`, and `_thumb.psd`.
4. **Input Photos**: Input photos inside member folders are automatically identified using filename keywords:
   * **Full Body**: `full`, `body`
   * **Headshot**: `head`, `face`, `helmet`, `bucket`, `off`
   * **Thumbnail**: `thumb`, `small`

---

## Installation & Setup

### Prerequisites

* **Python 3.9+**

### Install Dependencies

Run the following command to install required Python packages:

```bash
pip install Pillow rembg psd-tools pytoshop packbits numpy tqdm opencv-python-headless python-dotenv
```

---

## Usage

1. Configure `BASE_DIR` in `process_501st_batch.py` to point to your root work directory:
   ```python
   BASE_DIR = Path(r"Z:\501st")
   ```
2. Place member submission folders into `Z:\501st`.
3. Set your Hugging Face token securely via environment variables or a .env file
   ```bash
   (HF_TOKEN=your_token)
4. Execute the batch processor:
   ```bash
   python process_501st_batch.py
   ```

---

## Pipeline Workflow

```text
               ┌──────────────────────────────────┐
               │   Input Member Folder            │
               └──────────────────────────────────┘
                                │
                                ▼
               ┌──────────────────────────────────┐
               │ BiRefNet AI BG Removal & Cleanup │
               │ (rembg + OpenCV Morphological)   │
               └──────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
        ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│   FULL / THUMB Branch         │               │   HEAD Branch                 │
│ - Waist-Up Crop (52% Height)  │               │ - Bust Crop (35% Height)      │
│ - Waist-Fit (85% Width)       │               │ - Head-Fill Fit (88% Height)  │
└───────────────────────────────┘               └───────────────────────────────┘
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
                                ▼
               ┌──────────────────────────────────┐
               │ Head-Centered Horizontal Offset  │
               │ (get_head_center_x)              │
               └──────────────────────────────────┘
                                │
                                ▼
               ┌──────────────────────────────────┐
               │ Composite & PSD Layer Generation │
               │ (Overlay -> Subject -> BG)       │
               └──────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│ Progressive   │       │ Editable 3-   │       │ Interlaced    │
│ JPG Exports   │       │ Layer PSDs    │       │ 64-color GIF  │
└───────────────┘       └───────────────┘       └───────────────┘
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
                                ▼
               ┌──────────────────────────────────┐
               │   Create Member .ZIP Archive     │
               └──────────────────────────────────┘
```

---

## Output Deliverables

Output files are automatically generated and saved in a `complete/` subfolder inside each member directory:

```text
Z:\501st\TK8231\complete\
├── tk8231_full.jpg             # Optimized Progressive JPG (20KB - 50KB)
├── tk8231_head.jpg             # Optimized Progressive JPG (10KB - 40KB)
├── tk8231_thumb.gif            # Interlaced 64-Color GIF
├── tk8231_full_working.psd     # 3-Layer Editable Working PSD
├── tk8231_head_working.psd     # 3-Layer Editable Working PSD
├── tk8231_thumb_working.psd    # 3-Layer Editable Working PSD
└── tk8231.zip                  # Zip Archive containing final web images
```
