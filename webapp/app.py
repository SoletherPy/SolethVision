from flask import Flask, render_template_string, request
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import base64
import io
import os
import datetime
import json
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "..")))
from shared.modules.model import EnhancedFingerprintCNN
from shared.modules.utils import resource_path

# Performance monitoring
performance_log = []

def log_prediction(pred_class, confidence, filename):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "predicted_class": int(pred_class),
        "confidence": float(confidence),
        "filename": filename
    }
    performance_log.append(entry)
    with open(resource_path(__file__, "performance_log.json"), "w") as f:
        json.dump(performance_log, f, indent=2)

def load_performance_log():
    log_file = resource_path(__file__, "performance_log.json")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def get_performance_stats():
    if not performance_log:
        return {"total": 0, "avg_conf": 0, "high": 0, "medium": 0, "low": 0}
    total = len(performance_log)
    avg_conf = sum(p["confidence"] for p in performance_log) / total
    high = sum(1 for p in performance_log if p["confidence"] >= 0.9)
    medium = sum(1 for p in performance_log if 0.7 <= p["confidence"] < 0.9)
    low = sum(1 for p in performance_log if p["confidence"] < 0.7)
    return {"total": total, "avg_conf": avg_conf, "high": high, "medium": medium, "low": low}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")
print("🔎 Loading fingerprint_cnn_enhanced_best.pt...")
model = EnhancedFingerprintCNN(num_classes=4)
MODEL_WEIGHTS = resource_path(__file__, "../shared/models/fingerprint_cnn_enhanced_best.pt")
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.eval()
print("✅ Model loaded and ready.")

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((96,96)),
    transforms.ToTensor()
])

sol_paths = [
    [resource_path(__file__, f'../shared/fps/fps{i}/sol/sol{j}.png') for j in range(1, 5)]
    for i in range(1, 5)
]

def predict_solution(img):
    print("🖼️ Preprocessing input for CNN...")
    x = preprocess(img).unsqueeze(0)
    print(f"📊 Tensor shape: {x.shape}")
    with torch.no_grad():
        logits = model(x)
        print(f"🧠 Model logits: {logits.numpy()}")
        pred = logits.argmax(1).item()
        conf = torch.softmax(logits, dim=1)[0, pred].item()
        print(f"🎯 PREDICTED CLASS: {pred}, CONFIDENCE: {conf:.4f}")
    return pred, conf

app = Flask(__name__)

# Load existing performance log
performance_log.extend(load_performance_log())

@app.route("/")
def index():
    print("📄 Serving main web page.")
    stats = get_performance_stats()  # or get_perf_stats(), whichever you implemented
    stats_html = (
        f"<div style='color:#9f9; background:#1a3f22; border-radius:7px; max-width:400px; margin:10px auto 30px auto; padding:9px;'>"
        f"<b>📊 Total predictions:</b> {stats.get('total', stats.get('total_predictions', 0))}<br>"
        f"<b>📈 Average confidence:</b> {stats.get('avg_conf', stats.get('avg_confidence', 0))*100:.2f}%<br>"
        f"<b>🟢 High (≥0.9):</b> {stats.get('high', stats.get('high_confidence', 0))} "
        f"<b>🟡 Medium (0.7-0.9):</b> {stats.get('medium', stats.get('medium_confidence', 0))} "
        f"<b>🔴 Low (<0.7):</b> {stats.get('low', stats.get('low_confidence', 0))}"
        "</div>"
    )

    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>GTA V Fingerprint Solver (CNN)</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial,sans-serif;background:#111;color:#0f0;text-align:center;padding:20px;}
            h1 { color: #0f0; text-shadow: 0 0 10px #0f0;}
            #upload {background:#333;color:#0f0;border:2px solid #0f0;padding:15px 30px;font-size:16px;cursor:pointer;margin:30px auto;border-radius:5px;display:block;width:250px;}
            #result {margin:30px auto;background:#222;padding:20px;border:1px solid #0f0;border-radius:10px;max-width:600px;}
            .solution-piece {height:100px;margin:5px;border:2px solid #0f0;border-radius:5px;}
            .debug-info {color:#888;font-size:12px;margin-top:15px;white-space:pre-line;text-align:left;background:#111;padding:10px;border-radius:5px;}
        </style>
    </head>
    <body>
        <h1>🔍 GTA V Fingerprint Solver (CNN)</h1>
        ''' + stats_html + '''
        <p>📱 Take a photo of the fingerprint screen in GTA V</p>
        <input type="file" accept="image/*" capture="environment" id="upload">
        <div id="result"></div>
        <script>
        document.getElementById('upload').addEventListener('change', function(evt) {
            const file = evt.target.files[0];
            if (!file) return;
            document.getElementById('result').innerHTML = '<div style="color:#ff0;">🔄 Processing fingerprint...</div>';
            const reader = new FileReader();
            reader.onload = function(e) {
                fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: e.target.result })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        document.getElementById('result').innerHTML = '<div style="color:red">❌ Error: ' + data.error + '</div>';
                        return;
                    }
                    let img_html = "";
                    for (let i = 0; i < data.pieces.length; i++) {
                        img_html += `<img src="data:image/png;base64,${data.pieces[i]}" class="solution-piece" alt="Piece ${i+1}" title="Piece ${i+1}">`;
                    }
                    let debug_html = `<div class="debug-info">
🐛 <b>Debug Info:</b>
🎯 Model Prediction: Class ${data.class_id}
📊 Confidence: ${(data.confidence*100).toFixed(2)}%
📏 Original Size: ${data.original_size}
🔄 After Correction: ${data.corrected_size}
💾 Upload Saved: ${data.saved_upload}
📁 Solution Files:
${data.solution_files.join('\\n')}
                    </div>`;
                    document.getElementById('result').innerHTML =
                        `<div style="color:#0f0;">🧩 Solution pieces (in order):</div>${img_html}${debug_html}`;
                });
            };
            reader.readAsDataURL(file);
        });
        </script>
    </body>
    </html>
    ''')


@app.route("/process", methods=["POST"])
def process():
    try:
        print("📥 Received /process request.")
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        
        # Load image and get original size
        img = Image.open(io.BytesIO(img_data))
        original_size = f"{img.size[0]}x{img.size[1]}"
        print(f"📏 Original upload size: {original_size}")
        
        # ALWAYS CORRECT ORIENTATION FIRST - before saving and analyzing
        img = ImageOps.exif_transpose(img).convert('RGB')
        corrected_size = f"{img.size[0]}x{img.size[1]}"
        print(f"🔄 After orientation correction: {corrected_size}")

        # Save the correctly oriented image
        save_dir = resource_path(__file__, "uploads")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        filename = os.path.join(save_dir, f"upload_{timestamp}.jpg")
        img.save(filename)
        print(f"💾 Saved correctly oriented upload to: {filename}")

        # Predict using CNN on correctly oriented image
        class_id, confidence = predict_solution(img)

        # Log performance
        log_prediction(class_id + 1, confidence, filename)

        # Load solution pieces
        encoded_pieces = []
        solution_files = []
        for piece_path in sol_paths[class_id]:
            solution_files.append(piece_path)
            if os.path.exists(piece_path):
                with open(piece_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    encoded_pieces.append(encoded)
                print(f"✅ Loaded solution piece: {piece_path}")
            else:
                print(f"❌ ERROR: Solution piece not found: {piece_path}")
                return {'error': f'Solution piece not found: {piece_path}'}

        print(f"🚀 Returning solution for class {class_id+1}, confidence {confidence*100:.2f}%")
        print("=" * 60)
        
        return {
            'pieces': encoded_pieces,
            'class_id': class_id + 1,
            'confidence': confidence,
            'solution_files': solution_files,
            'saved_upload': filename,
            'original_size': original_size,
            'corrected_size': corrected_size
        }
    except Exception as e:
        print(f"💥 ERROR in /process: {e}")
        import traceback
        print(traceback.format_exc())
        return {'error': str(e)}

if __name__ == "__main__":
    print("🎮" + "=" * 58 + "🎮")
    print("🚀 Starting GTA V Fingerprint CNN Solver...")
    print("🔄 Automatic orientation correction enabled")
    print("📁 All uploads saved to uploads/ folder")
    print("📊 Performance tracking enabled")
    print("🌐 Server accessible at http://YOURIP:5000")
    print("🎮" + "=" * 58 + "🎮")
    app.run(host="0.0.0.0", port=5000, debug=True)
