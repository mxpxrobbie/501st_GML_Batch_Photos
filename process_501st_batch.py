import os
import sys
import warnings
import logging
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

# ---------------------------------------------------------------------------
# 1. Environment & Suppression Flags
# ---------------------------------------------------------------------------

# Enter your Hugging Face token here, inside the quotes. 
os.environ["HF_TOKEN"] = "ENTER TOKEN HERE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN_WARNING"] = "1"

warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*cache-system uses symlinks.*")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from transformers import logging as tf_logging
tf_logging.set_verbosity_error()
from transformers import pipeline

# Library imports
try:
    from psd_tools import PSDImage
    HAS_PSD_TOOLS = True
except ImportError:
    HAS_PSD_TOOLS = False

try:
    import pytoshop
    from pytoshop.user import nested_layers
    from pytoshop.enums import ColorMode, BlendMode, Compression
    HAS_PYTOSHOP = True
except ImportError:
    HAS_PYTOSHOP = False

# ---------------------------------------------------------------------------
# 2. Configuration & Base Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(r"Z:\501st")
FRAMESETS_DIR = BASE_DIR / "501st Framesets"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# 3. Pose Safety Scanner (CLIP)
# ---------------------------------------------------------------------------
POSE_CLASSIFIER = None

def get_pose_classifier():
    global POSE_CLASSIFIER
    if POSE_CLASSIFIER is None:
        POSE_CLASSIFIER = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch16"
        )
    return POSE_CLASSIFIER

def check_unapproved_pose(image_path: Path, confidence_threshold: float = 0.30) -> tuple[bool, str]:
    classifier = get_pose_classifier()
    
    unapproved_labels = [
        "a photo of a person pointing a blaster or gun directly at the camera lens",
        "aiming a weapon or prop forward at the viewer or camera",
        "a gun barrel or blaster muzzle pointed at the lens"
    ]
    approved_labels = [
        "a photo of a person holding a weapon pointed down at the floor",
        "a person standing at ease holding a prop to the side",
        "a holstered weapon or standing at attention"
    ]
    
    if "action" in image_path.name.lower():
        confidence_threshold = 0.20

    try:
        results = classifier(str(image_path), candidate_labels=unapproved_labels + approved_labels)
        top_label = results[0]["label"]
        unapproved_score = sum(r["score"] for r in results if r["label"] in unapproved_labels)
        
        print(f"    [Pose Check] {image_path.name}: Top Match = '{top_label[:45]}...' | Unapproved Score = {unapproved_score:.2%}")
        
        if top_label in unapproved_labels or unapproved_score >= confidence_threshold:
            return True, f"Unapproved action pose (weapon pointed at camera, score: {unapproved_score:.1%})"
            
    except Exception as e:
        print(f"    [Warning] Pose safety check error for {image_path.name}: {e}")
        
    return False, ""

# ---------------------------------------------------------------------------
# 4. Smart Content Cropping Helpers & Background Removal Engine
# ---------------------------------------------------------------------------
REMBG_SESSION = None

def get_rembg_session():
    global REMBG_SESSION
    if REMBG_SESSION is None:
        try:
            from rembg import new_session
            REMBG_SESSION = new_session("birefnet-general")
        except Exception:
            try:
                from rembg import new_session
                REMBG_SESSION = new_session("u2net")
            except Exception:
                REMBG_SESSION = None
    return REMBG_SESSION

def cleanup_isolated_artifacts(rgba_img: Image.Image) -> Image.Image:
    np_img = np.array(rgba_img)
    if np_img.shape[2] < 4:
        return rgba_img

    alpha = np_img[:, :, 3]
    _, binary_mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        main_mask = np.zeros_like(cleaned_mask)
        cv2.drawContours(main_mask, [main_contour], -1, 255, thickness=cv2.FILLED)
        np_img[:, :, 3] = cv2.bitwise_and(alpha, main_mask)

    return Image.fromarray(np_img)

def remove_background(image_path: Path) -> Image.Image:
    input_img = Image.open(image_path)
    try:
        from rembg import remove
        session = get_rembg_session()
        
        if session is not None:
            raw_cutout = remove(
                input_img,
                session=session,
                alpha_matting=False
            )
        else:
            raw_cutout = remove(input_img, alpha_matting=False)

        return cleanup_isolated_artifacts(raw_cutout)

    except Exception:
        return input_img.convert("RGBA")

def trim_transparent_padding(img: Image.Image) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img

def create_waist_up_crop(img: Image.Image) -> Image.Image:
    trimmed = trim_transparent_padding(img)
    w, h = trimmed.size
    cropped = trimmed.crop((0, 0, w, int(h * 0.55)))
    return trim_transparent_padding(cropped)

def create_shoulders_up_crop(img: Image.Image) -> Image.Image:
    trimmed = trim_transparent_padding(img)
    w, h = trimmed.size
    cropped = trimmed.crop((0, 0, w, int(h * 0.50)))
    return trim_transparent_padding(cropped)

def get_head_center_x(img_rgba: Image.Image) -> int:
    np_img = np.array(img_rgba)
    if np_img.shape[2] < 4:
        return img_rgba.width // 2
        
    alpha = np_img[:, :, 3]
    h, w = alpha.shape
    head_region = alpha[:int(h * 0.4), :]
    
    cols_with_alpha = np.where(head_region > 25)[1]
    if len(cols_with_alpha) > 0:
        return int(np.mean(cols_with_alpha))
    return w // 2

def categorize_member_photos(approved_images: list[Path]) -> tuple[Path | None, Path | None]:
    if not approved_images:
        return None, None
        
    head_keywords = ["helmet off", "helmetoff", "bucket off", "bucketoff", "headshot", "no helmet", "no-helmet", "no bucket", "face"]
    front_keywords = ["front", "full", "standing", "body", "main", "costume"]
    
    head_photo, front_photo = None, None
    
    for img in approved_images:
        name_lower = img.name.lower()
        if any(k in name_lower for k in head_keywords):
            head_photo = img
            break
            
    for img in approved_images:
        name_lower = img.name.lower()
        if any(k in name_lower for k in front_keywords):
            front_photo = img
            break
            
    if front_photo is None:
        for img in approved_images:
            if img != head_photo:
                front_photo = img
                break
                
    if front_photo is None:
        front_photo = approved_images[0]
        
    if head_photo is None:
        head_photo = front_photo
        
    return front_photo, head_photo

def get_member_id(folder_name: str) -> str:
    return folder_name.replace("-", "").replace(" ", "").lower()

def get_costume_prefix(folder_name: str) -> str:
    cleaned = folder_name.replace("-", "").strip()
    prefix = "".join([c for c in cleaned if c.isalpha()])
    return prefix.upper() if prefix else "TK"

# ---------------------------------------------------------------------------
# 5. Frameset & Overlay Engine
# ---------------------------------------------------------------------------
def find_frameset_template(view_type: str, costume_prefix: str = "") -> Path | None:
    if not FRAMESETS_DIR.exists():
        return None
        
    psd_files = [p for p in FRAMESETS_DIR.rglob("*") if p.suffix.lower() == ".psd"]
    if not psd_files:
        return None
        
    for psd in psd_files:
        name = psd.name.lower()
        if costume_prefix.lower() in name and view_type.lower() in name:
            return psd
            
    for psd in psd_files:
        if view_type.lower() in psd.name.lower():
            return psd
            
    return psd_files[0]

def generate_procedural_frame(width: int, height: int) -> tuple[Image.Image, Image.Image]:
    bg = Image.new("RGBA", (width, height), (25, 30, 36, 255))
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    bw = int(min(width, height) * 0.04)
    draw.rectangle([0, 0, width, height], outline=(192, 196, 200, 255), width=bw)
    draw.rectangle([bw + 2, bw + 2, width - bw - 2, height - bw - 2], outline=(120, 125, 130, 255), width=2)
    
    return bg, overlay

def load_frameset_layers(template_path: Path | None, target_w: int = 1000, target_h: int = 1500):
    if template_path and template_path.exists() and HAS_PSD_TOOLS:
        try:
            psd = PSDImage.open(template_path)
            w, h = psd.width, psd.height
            layers = [l for l in psd.descendants() if l.is_visible() and not l.is_group()]
            
            overlay_img, bg_img = None, None
            for l in layers:
                name = l.name.lower()
                if any(k in name for k in ["frame", "overlay", "border", "top", "501st"]):
                    overlay_img = l.topil()
                elif any(k in name for k in ["bg", "background", "back", "bottom", "paper"]):
                    bg_img = l.topil()
                    
            if overlay_img is None and len(layers) > 0:
                overlay_img = layers[0].topil()
            if bg_img is None and len(layers) > 0:
                bg_img = layers[-1].topil()
                
            bg_rgba = bg_img.convert("RGBA").resize((w, h)) if bg_img else Image.new("RGBA", (w, h), (30, 34, 40, 255))
            overlay_rgba = overlay_img.convert("RGBA").resize((w, h)) if overlay_img else Image.new("RGBA", (w, h), (0, 0, 0, 0))
            return bg_rgba, overlay_rgba, w, h
        except Exception as e:
            print(f"    [Frameset] Notice reading template ({e}). Using metallic fallback.")

    return generate_procedural_frame(target_w, target_h) + (target_w, target_h)

# ---------------------------------------------------------------------------
# 6. Uncompressed 3-Layer PSD Writer
# ---------------------------------------------------------------------------
def write_3layer_psd(bg_img: Image.Image, trooper_img: Image.Image, overlay_img: Image.Image, output_psd_path: Path, layer_offset=(0, 0)):
    w, h = overlay_img.size
    bg_rgba = bg_img.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
    overlay_rgba = overlay_img.convert("RGBA").resize((w, h), Image.Resampling.LANCZOS)
    trooper_rgba = trooper_img.convert("RGBA")

    off_x, off_y = layer_offset

    if HAS_PYTOSHOP:
        try:
            def build_layer(img, name, top=0, left=0):
                r, g, b, a = img.split()
                channels = {
                    -1: np.array(a, dtype=np.uint8),
                     0: np.array(r, dtype=np.uint8),
                     1: np.array(g, dtype=np.uint8),
                     2: np.array(b, dtype=np.uint8)
                }
                return nested_layers.Image(
                    name=name,
                    visible=True,
                    opacity=255,
                    group_id=0,
                    blend_mode=BlendMode.normal,
                    channels=channels,
                    top=top,
                    left=left
                )

            layer_top = build_layer(overlay_rgba, "Frame Overlay", top=0, left=0)
            layer_mid = build_layer(trooper_rgba, "Trooper Photo", top=off_y, left=off_x)
            layer_bot = build_layer(bg_rgba, "Background", top=0, left=0)

            psd_structure = nested_layers.nested_layers_to_psd(
                [layer_top, layer_mid, layer_bot],
                color_mode=ColorMode.rgb,
                compression=Compression.raw
            )
            with open(output_psd_path, "wb") as f:
                psd_structure.write(f)
            return
        except Exception as e:
            print(f"    [PSD Export Error] {e}")

# ---------------------------------------------------------------------------
# 7. Asset Generator
# ---------------------------------------------------------------------------
def generate_member_assets(member_folder: Path, front_photo: Path, head_photo: Path):
    complete_dir = member_folder / "complete"
    complete_dir.mkdir(parents=True, exist_ok=True)
    
    member_id = get_member_id(member_folder.name)
    costume_prefix = get_costume_prefix(member_folder.name)
    
    print(f"  -> Processing 501st Framed Package for: {member_id} ({costume_prefix})")

    processed_front = remove_background(front_photo)
    processed_head = remove_background(head_photo)
    
    full_crop = create_waist_up_crop(processed_front)
    thumb_crop = create_waist_up_crop(processed_front)
    head_crop = create_shoulders_up_crop(processed_head)

    # Deliverable paths
    full_jpg = complete_dir / f"{member_id}_full.jpg"
    full_psd = complete_dir / f"{member_id}_full_working.psd"
    head_jpg = complete_dir / f"{member_id}_head.jpg"
    head_psd = complete_dir / f"{member_id}_head_working.psd"
    thumb_gif = complete_dir / f"{member_id}_thumb.gif"
    thumb_psd = complete_dir / f"{member_id}_thumb_working.psd"
    zip_path = complete_dir / f"{member_id}.zip"

    # Templates
    full_template = find_frameset_template("full", costume_prefix)
    head_template = find_frameset_template("head", costume_prefix)
    thumb_template = find_frameset_template("thumb", costume_prefix)

    def build_view(view_type, source_cutout, template, jpg_path, psd_path, is_gif=False):
        bg, overlay, w, h = load_frameset_layers(template)
        
        if view_type == "head":
            top_margin = int(h * 0.10)
            target_h = h - top_margin
            aspect = source_cutout.width / source_cutout.height
            target_w = int(target_h * aspect)
            
            fit_trooper = source_cutout.resize((target_w, target_h), Image.Resampling.LANCZOS)
            
            head_x = get_head_center_x(fit_trooper)
            off_x = (w // 2) - head_x
            off_y = top_margin

            trooper_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            trooper_canvas.paste(fit_trooper, (off_x, off_y), fit_trooper)
            
            merged = Image.alpha_composite(Image.alpha_composite(bg, trooper_canvas), overlay).convert("RGB")
            write_3layer_psd(bg, fit_trooper, overlay, psd_path, layer_offset=(off_x, off_y))
        else:
            if view_type == "full":
                top_margin = int(h * 0.16)
                bottom_margin = int(h * 0.08)
                avail_w = int(w * 0.82)
            else: # thumb
                top_margin = int(h * 0.14)
                bottom_margin = int(h * 0.06)
                avail_w = int(w * 0.84)

            avail_h = h - top_margin - bottom_margin
            fit_trooper = ImageOps.contain(source_cutout.convert("RGBA"), (avail_w, avail_h), Image.Resampling.LANCZOS)
            
            head_x = get_head_center_x(fit_trooper)
            off_x = (w // 2) - head_x
            off_y = (h - bottom_margin) - fit_trooper.height

            trooper_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            trooper_canvas.paste(fit_trooper, (off_x, off_y), fit_trooper)
            
            merged = Image.alpha_composite(Image.alpha_composite(bg, trooper_canvas), overlay).convert("RGB")
            write_3layer_psd(bg, trooper_canvas, overlay, psd_path, layer_offset=(0, 0))
        
        if is_gif:
            merged.quantize(colors=256).save(jpg_path, "GIF", interlace=False)
        else:
            merged.save(jpg_path, "JPEG", quality=100, subsampling=0)

    # 1. Full View
    build_view("full", full_crop, full_template, full_jpg, full_psd)
    
    # 2. Headshot View
    build_view("head", head_crop, head_template, head_jpg, head_psd)
    
    # 3. Thumbnail View
    build_view("thumb", thumb_crop, thumb_template, thumb_gif, thumb_psd, is_gif=True)

    # 4. Final ZIP Package
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in [full_jpg, head_jpg, thumb_gif]:
            if f.exists():
                zipf.write(f, arcname=f.name)

    print(f"  -> Generated 7 framed deliverables in {complete_dir.relative_to(BASE_DIR)}")

# ---------------------------------------------------------------------------
# 8. Main Loop
# ---------------------------------------------------------------------------
def process_501st_batch(base_directory: Path = BASE_DIR):
    if not base_directory.exists():
        print(f"Error: Base directory '{base_directory}' not found.")
        return

    print(f"=== Starting 501st Member Photo Batch Processing ===")
    print(f"Base Directory: {base_directory}\n")

    summary_report = []
    member_folders = [f for f in base_directory.iterdir() if f.is_dir() and f.name != "501st Framesets"]

    for folder in member_folders:
        complete_dir = folder / "complete"
        member_id = get_member_id(folder.name)
        expected_zip = complete_dir / f"{member_id}.zip"
        
        if complete_dir.exists() and not expected_zip.exists():
            for stale_file in complete_dir.iterdir():
                try:
                    if stale_file.is_file():
                        stale_file.unlink()
                except Exception:
                    pass

        if expected_zip.exists():
            print(f"[SKIP] {folder.name}: Package ({expected_zip.name}) is complete.")
            continue

        images = [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
        if not images:
            continue

        print(f"\n[PROCESSING] Member Folder: {folder.name}")

        approved_images = []
        for img_path in images:
            is_unapproved, reason = check_unapproved_pose(img_path)
            if is_unapproved:
                print(f"  --> [FLAGGED]: {reason}")
                summary_report.append(f"{folder.name} / {img_path.name}: FLAGGED - {reason}")
            else:
                approved_images.append(img_path)

        if not approved_images:
            print(f"  --> [SKIP] No approved images remaining in {folder.name}")
            continue

        front_photo, head_photo = categorize_member_photos(approved_images)
        if not front_photo:
            print(f"  --> [ERROR] Could not determine front photo for {folder.name}")
            continue

        try:
            generate_member_assets(folder, front_photo, head_photo)
            summary_report.append(f"{folder.name}: SUCCESS (Full: {front_photo.name}, Head: {head_photo.name})")
        except Exception as e:
            print(f"  --> [ERROR] Processing failed: {e}")
            summary_report.append(f"{folder.name}: ERROR - {e}")

    report_path = base_directory / "batch_run_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("501st Batch Processing Execution Summary\n")
        f.write("========================================\n\n")
        f.write("\n".join(summary_report))

    print(f"\n=== Batch Execution Complete. Summary saved to {report_path.name} ===")

if __name__ == "__main__":
    process_501st_batch(BASE_DIR)