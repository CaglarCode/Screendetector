import cv2
import numpy as np

def is_anomalous(frame, threshold=100):
    """
    Analiz için görüntüde olağandışı renkler veya yazı olup olmadığını tespit eder.
    
    frame: OpenCV tarafından yakalanmış görüntü
    threshold: Yazı tespiti için piksel yoğunluğu eşik değeri
    
    Returns: bool
    """
    # Görüntüyü griye çevir
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yazı tespiti için görüntüye kenar algılama uygula
    edges = cv2.Canny(gray_frame, 50, 150)

    # Yoğun piksel miktarını hesapla
    pixel_density = np.sum(edges) / edges.size

    if pixel_density > threshold:
        return True  # Olağandışı içerik var

    return False

def process_video(video_path, output_path):
    """
    Videoyu işleyip olağandışı içerik tespit edilen karelerde ekranı karartır.

    video_path: Giriş videosunun yolu
    output_path: Çıktı videosunun yolu
    """
    cap = cv2.VideoCapture(video_path)
    
    # Video özelliklerini al
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Olağandışı içerik var mı kontrol et
        if is_anomalous(frame):
            # Kara ekran oluştur
            black_frame = np.zeros_like(frame)
            out.write(black_frame)
        else:
            out.write(frame)

    cap.release()
    out.release()
    print("Video işleme tamamlandı. Çıktı:", output_path)

# Örnek kullanım
input_video = "input_video.mp4"
output_video = "output_video_blackout.mp4"
process_video(input_video, output_video)
