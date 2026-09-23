import os
import re
import zipfile
import builtins
from pathlib import Path
from PIL import Image, ImageOps
from rembg import remove
from psd_tools import PSDImage
from tqdm import tqdm

# Fix for pytoshop internal PackBits bug
import packbits
builtins.packbits = packbits

# Layered PSD exporter library
import numpy as np
import pytoshop
from pytoshop.user import nested_layers

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(r"Z:\501st")
FRAMESETS_BASE_DIR = BASE_DIR / "501st Framesets"

# Folders inside Z:\501st to skip when looking for member folders
EXCLUDE_FOLDERS = {"501st Framesets", "complete", "templates", "temp"}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def extract_costume_prefix(folder_name: str) -> str:
    """Extracts leading letters from a member folder name (e.g., 'TK8231' -> 'TK')."""
    match = re.match(r"^([a-zA-Z]+)", folder_name.strip())
    if match:
        return match.group(1).upper()
    raise ValueError(f"Could not extract costume prefix from folder name: '{folder_name}'")


def find_frameset_folder(prefix: str) -> Path:
    """Finds the subfolder inside '501st Framesets' matching the costume prefix."""
    if not FRAMESETS_BASE_DIR.exists():
        raise FileNotFoundError(f"Framesets base directory not found at {FRAMESETS_BASE_DIR}")

    for subfolder in FRAMESETS_BASE_DIR.iterdir():
        if subfolder.is_dir():
            if subfolder.name.upper().startswith(prefix) or subfolder.name.upper() == prefix:
                return subfolder

    raise FileNotFoundError(
        f"No frameset folder found in '{FRAMESETS_BASE_DIR}' matching prefix '{prefix}'"
    )


def get_frameset_psd(frameset_dir: Path, image_type: str) -> Path:
    """Recursively searches inside frameset folder for PSD ending with '_{image_type}.psd'."""
    matches = list(frameset_dir.rglob(f"*_{image_type}.psd"))
    if not matches:
        raise FileNotFoundError(
            f"Could not find *_{image_type}.psd inside {frameset_dir} or any of its subfolders."
        )
    return matches[0]


def load_psd_layers(psd_path: Path):
    """Extracts Background (Layer 0) and Frame Overlay (Layers 1+) from a PSD file."""
    psd = PSDImage.open(psd_path)
    
    bg_layer = psd[0].topil().convert("RGBA")
    frame_overlay = Image.new("RGBA", psd.size, (0, 0, 0, 0))
    
    for layer in psd[1:]:
        if getattr(layer, "visible", True):
            layer_img = layer.topil()
            if layer_img:
                layer_img = layer_img.convert("RGBA")
                frame_overlay.paste(layer_img, (layer.left, layer.top), layer_img)
            
    return bg_layer, frame_overlay


def remove_background(image_path: Path) -> Image.Image:
    """Strips background from input photo using AI segmentation."""
    with open(image_path, "rb") as f:
        input_data = f.read()
    output_data = remove(input_data)
    from io import BytesIO
    return Image.open(BytesIO(output_data)).convert("RGBA")


def get_head_center_x(subject_img: Image.Image) -> float:
    """
    Calculates horizontal center X-coordinate of the head/face region 
    (top 35% of subject cutout) to ensure precise centering.
    """
    bbox = subject_img.getbbox()
    if not bbox:
        return subject_img.width / 2.0
        
    left, top, right, bottom = bbox
    head_region_height = max(int((bottom - top) * 0.35), 10)
    
    head_crop = subject_img.crop((left, top, right, top + head_region_height))
    head_bbox = head_crop.getbbox()
    
    if head_bbox:
        return left + (head_bbox[0] + head_bbox[2]) / 2.0
        
    return (left + right) / 2.0


def crop_waist_up(subject_img: Image.Image, height_ratio: float = 0.52) -> Image.Image:
    """Crops transparent RGBA subject from top of head to waist (~52% down)."""
    bbox = subject_img.getbbox()
    if not bbox:
        return subject_img
        
    left, top, right, bottom = bbox
    subject_width = right - left
    subject_height = bottom - top
    
    if subject_height / max(subject_width, 1) > 1.4:
        waist_bottom = top + int(subject_height * height_ratio)
        return subject_img.crop((left, top, right, waist_bottom))
        
    return subject_img.crop((left, top, right, bottom))


def crop_bust_shot(subject_img: Image.Image, height_ratio: float = 0.35) -> Image.Image:
    """Crops transparent RGBA subject to a bust view (chest/shoulders to top of head)."""
    bbox = subject_img.getbbox()
    if not bbox:
        return subject_img
        
    left, top, right, bottom = bbox
    subject_width = right - left
    subject_height = bottom - top
    
    if subject_height / max(subject_width, 1) > 1.4:
        bust_bottom = top + int(subject_height * height_ratio)
        return subject_img.crop((left, top, right, bust_bottom))
        
    return subject_img.crop((left, top, right, bottom))


def composite_photo(subject_img: Image.Image, bg_img: Image.Image, frame_overlay: Image.Image, fit_mode: str = "waist"):
    """
    Composites subject onto background and frame overlay.
    Uses head-center alignment to keep face dead-centered in frame.
    """
    canvas = bg_img.copy().convert("RGBA")
    
    if fit_mode == "head":
        # Target 88% of canvas height to fill frame window
        target_h = int(canvas.height * 0.88)
        h_ratio = target_h / max(subject_img.height, 1)
        
        new_h = target_h
        new_w = int(subject_img.width * h_ratio)
        
        subject_resized = subject_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        bottom_margin = int(canvas.height * 0.02) # 2% margin at bottom frame border
        
    else: # 'waist' mode for Full/Thumb photos
        target_w = int(canvas.width * 0.85)
        w_ratio = target_w / max(subject_img.width, 1)
        new_w = target_w
        new_h = int(subject_img.height * w_ratio)
        
        max_h = int(canvas.height * 0.80)
        if new_h > max_h:
            h_ratio = max_h / new_h
            new_h = max_h
            new_w = int(new_w * h_ratio)
            
        subject_resized = subject_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        bottom_margin = int(canvas.height * 0.05)
        
    # Align specifically by the head center rather than bounding box edge
    head_center_x = get_head_center_x(subject_resized)
    x_offset = int((canvas.width / 2.0) - head_center_x)
    y_offset = canvas.height - subject_resized.height - bottom_margin
        
    canvas.paste(subject_resized, (x_offset, y_offset), subject_resized)
    canvas.paste(frame_overlay, (0, 0), frame_overlay)
    
    return canvas, subject_resized, (x_offset, y_offset)


def save_layered_psd(bg_img: Image.Image, subject_img: Image.Image, frame_overlay: Image.Image, subject_offset: tuple, out_path: Path):
    """
    Saves an editable 3-layer Photoshop PSD file.
    Layer Order (Top to Bottom): Frame Overlay -> Trooper Photo -> Background
    """
    width, height = bg_img.size

    subject_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    subject_canvas.paste(subject_img, subject_offset, subject_img)

    def pil_to_psd_layer(pil_img: Image.Image, name: str):
        arr = np.array(pil_img.convert("RGBA"))
        channels = {
            0: arr[:, :, 0],   # Red
            1: arr[:, :, 1],   # Green
            2: arr[:, :, 2],   # Blue
            -1: arr[:, :, 3]   # Alpha
        }
        return nested_layers.Image(channels=channels, name=name)

    layer_bg = pil_to_psd_layer(bg_img, "Background")
    layer_subject = pil_to_psd_layer(subject_canvas, "Trooper Photo")
    layer_frame = pil_to_psd_layer(frame_overlay, "Frame Overlay")

    psd_layers = [layer_frame, layer_subject, layer_bg]
    psd_file = nested_layers.nested_layers_to_psd(psd_layers, color_mode=pytoshop.enums.ColorMode.rgb)

    with open(out_path, "wb") as f:
        psd_file.write(f)


def save_optimized_jpg(img: Image.Image, out_path: Path, min_kb: int, max_kb: int):
    """Saves progressive JPG with quality auto-tuning starting from 95% downward."""
    rgb_img = img.convert("RGB")
    
    for quality in range(95, 25, -5):
        rgb_img.save(out_path, "JPEG", quality=quality, progressive=True)
        size_kb = out_path.stat().st_size / 1024
        if min_kb <= size_kb <= max_kb:
            return
        
    rgb_img.save(out_path, "JPEG", quality=95, progressive=True)


def save_optimized_gif(img: Image.Image, out_path: Path, colors: int = 64):
    """Saves interlaced GIF thumbnail with color quantization."""
    gif_img = img.convert("RGB").quantize(colors=colors, dither=Image.Dither.NONE)
    gif_img.save(out_path, "GIF", interlace=True)


def classify_photos(photo_paths: list[Path]):
    """Categorizes member photos into (full, head, thumb)."""
    full, head, thumb = None, None, None
    remaining = []

    for p in photo_paths:
        name = p.stem.lower()
        if "full" in name or "body" in name:
            full = p
        elif "head" in name or "face" in name or "helmet" in name or "bucket" in name or "off" in name:
            head = p
        elif "thumb" in name or "small" in name:
            thumb = p
        else:
            remaining.append(p)

    if not full and remaining:
        full = remaining.pop(0)
    if not head and remaining:
        head = remaining.pop(0)
    if not thumb and remaining:
        thumb = remaining.pop(0)

    if not thumb:
        thumb = full

    return full, head, thumb

# ==========================================
# MAIN BATCH PROCESSOR
# ==========================================

def process_member_folder(member_dir: Path):
    complete_dir = member_dir / "complete"
    
    if complete_dir.exists():
        print(f"\nSkipping [{member_dir.name}]: 'complete' folder already exists.")
        return

    member_id = member_dir.name.lower().strip() # e.g. tk8231
    prefix = extract_costume_prefix(member_dir.name)
    frameset_dir = find_frameset_folder(prefix)

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".tif"}
    raw_photos = [
        p for p in member_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    if not raw_photos:
        return

    print(f"\nProcessing Member Folder: [{member_dir.name}] (Prefix: '{prefix}')")

    full_src, head_src, thumb_src = classify_photos(raw_photos)

    with tqdm(total=8, desc=f"  {member_id.upper()}", unit="step", bar_format="{desc}: |{bar:30}| {percentage:3.0f}% [{postfix}]") as pbar:
        
        complete_dir.mkdir(exist_ok=True)

        # Step 1: Process FULL - BG Removal & Waist-Up Crop
        pbar.set_postfix_str("1/8 AI BG Removal & Waist-Up Crop (FULL)")
        full_psd_path = get_frameset_psd(frameset_dir, "full")
        bg_full, overlay_full = load_psd_layers(full_psd_path)
        
        full_no_bg = remove_background(full_src)
        full_waist_up = crop_waist_up(full_no_bg, height_ratio=0.52)
        pbar.update(1)

        # Step 2: Process FULL - Composite & Save (Waist Fit)
        pbar.set_postfix_str("2/8 Compositing FULL JPG & PSD")
        full_comp, full_subj_resized, full_offset = composite_photo(full_waist_up, bg_full, overlay_full, fit_mode="waist")
        full_out = complete_dir / f"{member_id}_full.jpg"
        save_optimized_jpg(full_comp, full_out, min_kb=20, max_kb=50)
        full_psd_out = complete_dir / f"{member_id}_full_working.psd"
        save_layered_psd(bg_full, full_subj_resized, overlay_full, full_offset, full_psd_out)
        pbar.update(1)

        # Step 3: Process HEAD - BG Removal & Bust Crop
        pbar.set_postfix_str("3/8 AI BG Removal & Bust Crop (HEAD)")
        head_psd_path = get_frameset_psd(frameset_dir, "head")
        bg_head, overlay_head = load_psd_layers(head_psd_path)
        
        head_no_bg = remove_background(head_src)
        head_bust = crop_bust_shot(head_no_bg, height_ratio=0.35)
        pbar.update(1)

        # Step 4: Process HEAD - Composite & Save (Head Fill Fit)
        pbar.set_postfix_str("4/8 Compositing HEAD JPG & PSD")
        head_comp, head_subj_resized, head_offset = composite_photo(head_bust, bg_head, overlay_head, fit_mode="head")
        head_out = complete_dir / f"{member_id}_head.jpg"
        save_optimized_jpg(head_comp, head_out, min_kb=10, max_kb=40)
        head_psd_out = complete_dir / f"{member_id}_head_working.psd"
        save_layered_psd(bg_head, head_subj_resized, overlay_head, head_offset, head_psd_out)
        pbar.update(1)

        # Step 5: Process THUMB - Reuses Waist-Up Cutout
        pbar.set_postfix_str("5/8 Preparing THUMB (Reusing Waist-Up Cutout)")
        thumb_psd_path = get_frameset_psd(frameset_dir, "thumb")
        bg_thumb, overlay_thumb = load_psd_layers(thumb_psd_path)
        thumb_no_bg = full_waist_up
        pbar.update(1)

        # Step 6: Process THUMB - Composite & Save (Waist Fit)
        pbar.set_postfix_str("6/8 Compositing THUMB GIF & PSD")
        thumb_comp, thumb_subj_resized, thumb_offset = composite_photo(thumb_no_bg, bg_thumb, overlay_thumb, fit_mode="waist")
        thumb_out = complete_dir / f"{member_id}_thumb.gif"
        save_optimized_gif(thumb_comp, thumb_out, colors=64)
        thumb_psd_out = complete_dir / f"{member_id}_thumb_working.psd"
        save_layered_psd(bg_thumb, thumb_subj_resized, overlay_thumb, thumb_offset, thumb_psd_out)
        pbar.update(1)

        # Step 7: Package into ZIP Archive
        pbar.set_postfix_str("7/8 Creating .ZIP Archive")
        zip_out = complete_dir / f"{member_id}.zip"
        with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(full_out, full_out.name)
            zipf.write(head_out, head_out.name)
            zipf.write(thumb_out, thumb_out.name)
        pbar.update(1)

        # Step 8: Complete
        pbar.set_postfix_str("8/8 Complete!")
        pbar.update(1)

    print(f"  [SUCCESS] Completed {member_id.upper()} -> Output folder: {complete_dir}")


def run_batch():
    print(f"Scanning base directory: {BASE_DIR}\n")
    for item in BASE_DIR.iterdir():
        if item.is_dir() and item.name not in EXCLUDE_FOLDERS:
            try:
                process_member_folder(item)
            except Exception as e:
                print(f"  [ERROR] Skipping {item.name}: {e}")

if __name__ == "__main__":
    run_batch()