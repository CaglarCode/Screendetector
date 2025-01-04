import cv2
import numpy as np
from ultralytics import YOLO  # RSOD detection model

class RealTimeStreamProcessor:
    def __init__(self, input_stream, model_path):
        self.input_stream = input_stream
        self.model = YOLO(model_path)

    def is_anomalous(self, frame, precision=0.8):
        """
     
       
        precision: Algılama doğruluğu

        Returns: bool
        """
        results = self.model.predict(source=frame, conf=precision)
        for r in results:
            if r.boxes:  # Eğer bir obje algılandıysa
                return True
        return False

    def process_stream(self):
        """
        CDN akışını okuyup kara ekran gerektiğinde mesaj yazar.
        """
        print("Başlangıç: CDN URL üzerindeki yazı veya kod algılanırsa black frame işlemine geçilecektir.")

        cap = cv2.VideoCapture(self.input_stream)

        if not cap.isOpened():
            print("Akışa bağlanılamadı.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Akış kesildi.")
                break

            # Algılama kontrolü
            if self.is_anomalous(frame):
                print("Anomali tespit edildi: Black frame uygulanması gerekmektedir.")
            else:
                print("Normal kare işlendi.")

        cap.release()
        print("Yayın işleme tamamlandı.")

# Örnek kullanım
if __name__ == "__main__":
    input_stream = "https://dai.google.com/linear/hls/event/GxrCGmwST0ixsrc_QgB6qw/master.m3u8"  # CDN URL
    model_path = "yolov8n.pt"  # Pre-trained model path

    processor = RealTimeStreamProcessor(input_stream, model_path)
    processor.process_stream()

