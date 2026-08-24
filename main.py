import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

try:
    from plyer import filechooser
except ImportError:
    filechooser = None

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite import Interpreter
    except ImportError:
        Interpreter = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.tflite")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
OGHAM_FONT_PATH = os.path.join(BASE_DIR, "NotoSansOgham-Regular.ttf")


class TFLiteOghamPredictor:
    def __init__(self):
        if Interpreter is None:
            raise RuntimeError("TFLite runtime is not available.")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("best.tflite missing.")
        
        with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
            self.class_names = json.load(f)
            
        self.interpreter = Interpreter(model_path=MODEL_PATH, num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]
        self.input_h, self.input_w = map(int, self.input_info["shape"][1:3])

    def process_image(self, pil_img, confidence=0.25, iou=0.5):
        tensor, scale, left, top = self._prepare(pil_img)
        self.interpreter.set_tensor(self.input_info["index"], tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_info["index"])
        
        qscale, qzero = self.output_info["quantization"]
        if qscale:
            output = (output.astype(np.float32) - qzero) * qscale
        else:
            output = output.astype(np.float32)

        detections = self._decode(output, pil_img.width, pil_img.height, scale, left, top, confidence)
        
        # الرسم باستخدام PIL بدلاً من OpenCV
        annotated = pil_img.copy()
        draw = ImageDraw.Draw(annotated)
        for d in detections:
            draw.rectangle([d['x1'], d['y1'], d['x2'], d['y2']], outline=(30, 210, 100), width=3)

        if not detections:
            return annotated, "", "No Ogham characters detected."
        return annotated, *self._make_text(detections)

    def _prepare(self, img):
        w, h = img.size
        scale = min(self.input_w / w, self.input_h / h)
        nw, nh = round(w * scale), round(h * scale)
        
        resized = img.resize((nw, nh), PILImage.BILINEAR)
        canvas = PILImage.new("RGB", (self.input_w, self.input_h), (114, 114, 114))
        left, top = (self.input_w - nw) // 2, (self.input_h - nh) // 2
        canvas.paste(resized, (left, top))
        
        arr = np.array(canvas, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0), scale, left, top

    def _decode(self, output, orig_w, orig_h, scale, left, top, confidence):
        data = np.squeeze(output)
        if data.ndim != 2:
            return []
        if data.shape[0] < data.shape[1]:
            data = data.T

        boxes, scores, ids = [], [], []
        for row in data:
            if len(row) < 5:
                continue
            class_id = int(np.argmax(row[4:]))
            score = float(row[4 + class_id])
            if score < confidence or str(class_id) not in self.class_names:
                continue
            
            cx, cy, w, h = row[:4]
            x1 = max(0, (cx - w / 2 - left) / scale)
            y1 = max(0, (cy - h / 2 - top) / scale)
            x2 = min(orig_w, (cx + w / 2 - left) / scale)
            y2 = min(orig_h, (cy + h / 2 - top) / scale)
            
            if x2 > x1 and y2 > y1:
                boxes.append({"character": self.class_names[str(class_id)], "confidence": score, 
                              "x1": x1, "y1": y1, "x2": x2, "y2": y2, 
                              "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2, "height": y2 - y1})
        return boxes

    @staticmethod
    def _make_text(detections):
        detections.sort(key=lambda item: item["center_y"])
        avg_h = sum(item["height"] for item in detections) / len(detections) * 0.5
        lines = []
        for item in detections:
            for line in lines:
                if abs(item["center_y"] - np.mean([e["center_y"] for e in line])) < avg_h:
                    line.append(item)
                    break
            else:
                lines.append([item])
        
        final, details = [], []
        for i, line in enumerate(lines, 1):
            line.sort(key=lambda item: item["center_x"])
            final.append("".join(item["character"] for item in line))
            details.append(f"--- Line {i} ---")
            for item in line:
                details.append(f"Char: {item['character']} | Conf: {item['confidence']:.2f}")
        return "\n".join(final), "\n".join(details)


class OghamApp(App):
    def build(self):
        self.predictor = None
        self.current_img = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.img_display = Image(allow_stretch=True, keep_ratio=True)
        layout.add_widget(self.img_display)

        self.details_label = Label(text="Loading Model...", size_hint_y=0.2)
        layout.add_widget(ScrollView(size_hint_y=0.2, child=self.details_label))

        font = OGHAM_FONT_PATH if os.path.exists(OGHAM_FONT_PATH) else None
        self.final_label = Label(text="RESULT: -", font_name=font, font_size="20sp", size_hint_y=0.2)
        layout.add_widget(self.final_label)

        btn_box = BoxLayout(size_hint_y=0.15, spacing=10)
        self.btn_select = Button(text="Select Image", on_release=self.choose_image)
        self.btn_scan = Button(text="Scan Ogham", on_release=self.scan_image, disabled=True)
        btn_box.add_widget(self.btn_select)
        btn_box.add_widget(self.btn_scan)
        layout.add_widget(btn_box)

        self.executor.submit(self._init_model)
        return layout

    def _init_model(self):
        try:
            self.predictor = TFLiteOghamPredictor()
            Clock.schedule_once(lambda _: setattr(self.details_label, 'text', "Model Ready."))
            Clock.schedule_once(lambda _: setattr(self.btn_scan, 'disabled', False))
        except Exception as e:
            Clock.schedule_once(lambda _: setattr(self.details_label, 'text', f"Error: {e}"))

    def choose_image(self, _):
        if filechooser:
            filechooser.open_file(on_selection=self.on_file_selected)

    def on_file_selected(self, selection):
        if selection:
            self.current_img = PILImage.open(selection[0])
            self.update_display(self.current_img)
            self.details_label.text = "Image Loaded."

    def update_display(self, pil_img):
        data = pil_img.convert("RGBA").tobytes()
        texture = Texture.create(size=pil_img.size, colorfmt="rgba")
        texture.blit_buffer(data, colorfmt="rgba", bufferfmt="ubyte")
        texture.flip_vertical()
        self.img_display.texture = texture

    def scan_image(self, _):
        if self.current_img and self.predictor:
            self.details_label.text = "Scanning..."
            self.executor.submit(self._do_scan)

    def _do_scan(self):
        res_img, text, details = self.predictor.process_image(self.current_img)
        Clock.schedule_once(lambda _: self.update_display(res_img))
        Clock.schedule_once(lambda _: setattr(self.final_label, 'text', f"RESULT:\n{text}"))
        Clock.schedule_once(lambda _: setattr(self.details_label, 'text', details))

if __name__ == "__main__":
    OghamApp().run()