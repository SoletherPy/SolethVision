import sys
import json
import time
import keyboard
import numpy as np
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtWidgets import QApplication, QWidget
import mss
import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from shared.modules.model import EnhancedFingerprintCNN
from shared.modules.utils import resource_path

COMPARE_SIZE = (64, 64)
DEFAULT_SETTINGS = {
    "model_weights": "../shared/models/fingerprint_cnn_enhanced_best.pt",
    "roi_file": "./config/ROIs/fingerprint_rois.json",
    "fps_folder": "../shared/fps",
    "detection_key": "f",
    "clear_key": "c",
    "quit_key": "q"
}

def load_settings(settings_file="config/settings.json"):
    settings_file = resource_path(__file__, settings_file)
    if not os.path.exists(settings_file):
        print(f"📂 No settings file found at '{settings_file}'. Creating a template for you...")
        with open(settings_file, 'w') as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2)
        print(f"✅ Created '{settings_file}' with default settings.")
        print(f"📝 Please edit '{settings_file}' to match your file paths, then re-run this program.")
        sys.exit(0)
    else:
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
            required_keys = list(DEFAULT_SETTINGS.keys())
            missing_keys = [key for key in required_keys if key not in settings]
            if missing_keys:
                print(f"❌ Settings file is missing required keys: {', '.join(missing_keys)}")
                print(f"📝 Please add these keys to '{settings_file}' or delete the file to recreate it.")
                sys.exit(1)
            print(f"⚙️  Settings loaded successfully from '{settings_file}'")
            return settings
        except json.JSONDecodeError as e:
            print(f"❌ Settings file contains invalid JSON: {e}")
            print(f"💡 Hint: Delete '{settings_file}' to re-create a blank template.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error loading settings: {e}")
            print(f"💡 Hint: Delete '{settings_file}' to re-create a blank template.")
            sys.exit(1)

class Overlay(QWidget):
    def __init__(self, sol_boxes, watermark_text="SolethSight GTAV Overlay", watermark_pos="topleft"):
        super().__init__()
        self.screen = QApplication.primaryScreen().geometry()
        self.setGeometry(self.screen)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.sol_boxes = sol_boxes
        self.highlights = []
        self.font = QFont('Arial', 18)
        self.visible = True

        self.watermark_text = watermark_text
        self.watermark_font = QFont('Arial', 16, QFont.Bold)
        self.watermark_color = QColor(255, 255, 255, 160)
        self.watermark_pos = watermark_pos
        self.margin = 14

        self.show()

    def set_highlights(self, highlight_idxs):
        self.highlights = highlight_idxs
        self.update()

    def clear(self):
        self.highlights = []
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.visible and self.highlights:
            painter.setPen(QColor(0,255,0,180))
            painter.setFont(self.font)
            for idx in self.highlights:
                if idx < len(self.sol_boxes):
                    box = self.sol_boxes[idx]
                    r = QRect(box["x"], box["y"], box["w"], box["h"])
                    painter.drawRect(r)
                    painter.drawText(box["x"], box["y"]-12, f"SOL {idx+1}")

        painter.setFont(self.watermark_font)
        painter.setPen(self.watermark_color)
        size = painter.fontMetrics().size(Qt.TextSingleLine, self.watermark_text)
        if self.watermark_pos == "topleft":
            x = self.margin
        elif self.watermark_pos == "topright":
            x = self.width() - size.width() - self.margin
        else:
            x = self.margin
        y = size.height() + self.margin
        painter.drawText(x, y, self.watermark_text)

def get_scaled_rois(screen_w, screen_h, roi_file):
    with open(roi_file) as f:
        rois = json.load(f)
    sol_rel = [sol['relative'] for sol in rois['solutions']]
    boxes = [{"x": int(r["x"]*screen_w), "y": int(r["y"]*screen_h),
              "w": int(r["w"]*screen_w), "h": int(r["h"]*screen_h)} for r in sol_rel]
    fp = rois["fingerprint"]["relative"]
    fp_box = {"x": int(fp["x"]*screen_w), "y": int(fp["y"]*screen_h),
              "w": int(fp["w"]*screen_w), "h": int(fp["h"]*screen_h)}
    return fp_box, boxes

def run_inference(fp_crop, model, transform, device):
    img = Image.fromarray(fp_crop)
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        conf, cls = torch.max(probs, dim=0)
    return int(cls), float(conf)

def load_reference_solutions(fps_folder, class_id):
    class_path = os.path.join(fps_folder, f"fps{class_id+1}", "sol")
    refs = []
    for i in range(1, 5):
        ref_path = os.path.join(class_path, f"sol{i}.png")
        if os.path.exists(ref_path):
            ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
            refs.append(ref_img if ref_img is not None else None)
        else:
            refs.append(None)
    while len(refs) < 4:
        refs.append(None)
    return refs[:4]

def preprocess_for_match(img, out_size=COMPARE_SIZE):
    img = cv2.resize(img, out_size, interpolation=cv2.INTER_AREA)
    img = cv2.equalizeHist(img)
    return img

def main():
    app = QApplication(sys.argv)
    screen = app.primaryScreen().geometry()
    sw, sh = screen.width(), screen.height()
    fingerprint_box, sol_boxes = get_scaled_rois(sw, sh, ROI_FILE)
    print(f"Loaded {len(sol_boxes)} solution boxes from ROI file")
    overlay = Overlay(sol_boxes)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    model = EnhancedFingerprintCNN(num_classes=4)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    print(
    f"🟩 Overlay running!\n"
    f"👉 Press [{DETECTION_KEY}] 🟢 to detect solutions\n"
    f"👉 Press [{CLEAR_KEY}] 🧹 to clear overlay\n"
    f"👉 Press [{QUIT_KEY}] ❌ to quit"
    )
    last_f_time = 0
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while True:
            if keyboard.is_pressed(QUIT_KEY): 
                print(f"\n❌ [{QUIT_KEY}] pressed! Exiting overlay. Goodbye 👋")
                break
            if keyboard.is_pressed(DETECTION_KEY) and time.time() - last_f_time > 0.5:
                print("\n🔬 --- Starting fingerprint analysis ---")
                screen_np = np.array(sct.grab(monitor))[:, :, :3]
                fp = fingerprint_box
                fp_img = screen_np[fp["y"]:fp["y"]+fp["h"], fp["x"]:fp["x"]+fp["w"]]
                fp_img_pil = cv2.cvtColor(fp_img, cv2.COLOR_BGR2RGB)
                class_id, conf = run_inference(fp_img_pil, model, transform, device)
                print(f"🧬 Predicted fingerprint class: {class_id} (confidence: {conf:.3f})")
                refs = load_reference_solutions(FPS_FOLDER, class_id)

                highlights = []
                for ri, ref in enumerate(refs):
                    if ref is None: continue
                    ref_proc = preprocess_for_match(ref)
                    best_score = -1
                    best_idx = None
                    for i, box in enumerate(sol_boxes):
                        box_img = screen_np[box["y"]:box["y"]+box["h"], box["x"]:box["x"]+box["w"]]
                        box_gray = cv2.cvtColor(box_img, cv2.COLOR_BGR2GRAY)
                        box_proc = preprocess_for_match(box_gray)
                        ssim_score = ssim(ref_proc, box_proc)
                        if ssim_score > best_score:
                            best_score = ssim_score
                            best_idx = i
                    if best_idx is not None and best_idx not in highlights:
                        highlights.append(best_idx)
                        print(f"✅  ==> Box {best_idx+1} selected for Solution {ri+1} (SSIM {best_score:.3f})")
                print(f"✨ Highlighting {len(highlights)} solution(s): {', '.join(str(i+1) for i in highlights)}")
                overlay.set_highlights(highlights)
                last_f_time = time.time()
            if keyboard.is_pressed(CLEAR_KEY):
                print("🧹 Overlay cleared")
                overlay.clear()
            app.processEvents(); time.sleep(0.01)
    sys.exit()

if __name__ == "__main__":
    settings = load_settings()
    MODEL_WEIGHTS = resource_path(__file__, settings["model_weights"])
    ROI_FILE = resource_path(__file__, settings["roi_file"])
    FPS_FOLDER = resource_path(__file__, settings["fps_folder"])
    DETECTION_KEY = settings["detection_key"]
    CLEAR_KEY = settings["clear_key"]
    QUIT_KEY = settings["quit_key"]

    for file_path in [MODEL_WEIGHTS, ROI_FILE]:
        if not os.path.exists(file_path):
            print(f"❌ Required file not found: {file_path}")
            print("📝 Please check your settings.json and file locations.")
            sys.exit(1)
    
    if not os.path.exists(FPS_FOLDER):
        print(f"❌ Reference folder not found: {FPS_FOLDER}")
        print("📝 Please check your settings.json and folder location.")
        sys.exit(1)

    main()
