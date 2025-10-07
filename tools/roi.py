import cv2
import json
import os
import argparse
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from shared.modules.utils import resource_path

OVERLAY_ROI_FILE = resource_path(__file__, "../overlay/config/ROIs/fingerprint_rois.json")
DEFAULT_EXAMPLE = resource_path(__file__, "screenshot/exampleScreen.png")

class MultiROISelector:
    def __init__(self, image_path, n_solutions=8, output_path=None):
        image_path = resource_path(__file__, image_path)
        output_path = resource_path(__file__, output_path) if output_path else OVERLAY_ROI_FILE
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File '{image_path}' not found")
        self.image_path = image_path  # Store the image path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image '{image_path}'")
        self.orig_image = self.image.copy()
        self.image_height, self.image_width = self.image.shape[:2]
        self.n_solutions = n_solutions
        self.output_path = output_path or OVERLAY_ROI_FILE
        self.boxes = []
        self.fingerprint_box = None
        self.current_start = None
        self.current_end = None
        self.mode = "fingerprint"
        self.solution_idx = 0

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_start = (x, y)
            self.current_end = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.current_start:
                self.current_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.current_start:
                x1, y1 = self.current_start
                x2, y2 = x, y
                x0, y0 = min(x1, x2), min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)

                rel = {
                    "x": x0 / self.image_width,
                    "y": y0 / self.image_height,
                    "w": w / self.image_width,
                    "h": h / self.image_height
                }

                abs_box = {"x": x0, "y": y0, "w": w, "h": h}
                box = {"absolute": abs_box, "relative": rel}
                if self.mode == "fingerprint":
                    self.fingerprint_box = box
                    self.mode = "solutions"
                    print(f"✅ Fingerprint region marked!")
                elif self.mode == "solutions":
                    self.boxes.append(box)
                    self.solution_idx += 1
                    print(f"✅ Solution {self.solution_idx} marked!")
                self.current_start = None
                self.current_end = None

    def run(self):
        print(f"🎯 GTA V Fingerprint ROI Calibration Tool")
        print("=" * 50)
        print(f"🖼️  Loading image: {os.path.basename(self.image_path)}")  # Fixed: use self.image_path
        print(f"📏 Image size: {self.image_width}x{self.image_height}")
        print(f"🎯 Need to mark: 1 fingerprint + {self.n_solutions} solutions")
        print(f"📂 Output will save to: {self.output_path}")
        print("🖱️  Click and drag to select regions. Press 'r' to reset, 's' to save, 'q' to quit.\n")
        
        cv2.namedWindow("Mark ROIs", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Mark ROIs", self.mouse_callback)
        
        while True:
            display = self.orig_image.copy()
            # Draw approved ROIs
            if self.fingerprint_box:
                a = self.fingerprint_box['absolute']
                cv2.rectangle(display, (a['x'], a['y']), (a['x']+a['w'], a['y']+a['h']), (0,255,0), 2)
                cv2.putText(display, 'Fingerprint', (a['x'], max(0, a['y']-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            for i, box in enumerate(self.boxes):
                a = box['absolute']
                cv2.rectangle(display, (a['x'], a['y']), (a['x']+a['w'], a['y']+a['h']), (255,0,0), 2)
                cv2.putText(display, f'SOL {i+1}', (a['x'], max(0, a['y']-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            # Draw current drag box
            if self.current_start and self.current_end:
                cv2.rectangle(display, self.current_start, self.current_end, (0, 255, 255), 2)
            # Status text
            if self.mode == "fingerprint":
                msg = "Draw box around LARGE FINGERPRINT (Left click, drag, release)"
            elif self.solution_idx < self.n_solutions:
                msg = f"Draw box for SOLUTION {self.solution_idx+1} (Left click, drag, release)"
            else:
                msg = "Press 's' to save, 'q' to quit"
            cv2.putText(display, msg, (10, display.shape[0]-16), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Mark ROIs", display)
            key = cv2.waitKey(1) & 0xFF
            # Save and quit logic
            if key == ord('q'):
                print("❌ Quit without saving.")
                break
            elif key == ord('s') and self.fingerprint_box and self.solution_idx == self.n_solutions:
                self.save_rois()
                print("💾 ROIs saved successfully!")
                break
            elif key == ord('r'):  # Reset
                self.fingerprint_box = None
                self.boxes = []
                self.mode = "fingerprint"
                self.solution_idx = 0
                print("🔄 Reset all ROIs")
        cv2.destroyAllWindows()
        return self.fingerprint_box, self.boxes

    def save_rois(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out = {
            "image_size": {"width": self.image_width, "height": self.image_height},
            "fingerprint": self.fingerprint_box,
            "solutions": self.boxes
        }
        with open(self.output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"📂 ROIs saved to: {self.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🎯 GTA V ROI Calibration Tool - Select a screenshot for ROI marking.",
        epilog="Examples:\n"
               "  python roi.py\n"
               "  python roi.py --image my_screenshot.png\n"
               "  python roi.py --image screenshot.png --output custom_rois.json",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i", "--image", 
        type=str, 
        default=DEFAULT_EXAMPLE,
        help="🖼️  Path to fingerprint minigame screenshot (default: tools/screenshot/exampleScreen.png)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=OVERLAY_ROI_FILE,
        help="📂 Output path for ROI JSON file (default: ../overlay/config/ROIs/fingerprint_rois.json)"
    )
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    output_path = os.path.abspath(args.output)
    
    print(f"📂 Using screenshot: {resource_path(__file__, image_path)}")
    print(f"💾 Will save ROIs to: {resource_path(__file__, output_path)}")

    if not os.path.exists(resource_path(__file__, image_path)):
        print(f"❌ Screenshot not found: {resource_path(__file__, image_path)}")
        print("📝 Please provide a valid screenshot with --image or place one in tools/screenshot/")
        exit(1)

    selector = MultiROISelector(image_path, output_path=output_path)
    fingerprint_box, solution_boxes = selector.run()

    if fingerprint_box and solution_boxes:
        print(f"\n🎉 Calibration completed!")
        print(f"📊 Fingerprint region: {fingerprint_box['relative']}")
        for i, sol in enumerate(solution_boxes):
            print(f"📊 Solution {i+1}: {sol['relative']}")
        print("✨ Your overlay app will now use the new calibration!")
    else:
        print("\n⚠️  Calibration incomplete or cancelled.")