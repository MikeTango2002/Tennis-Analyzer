import cv2

def read_video(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:

        ret, frame = cap.read()
        if not ret:
            #non ci sono più frame da leggere
            break
        frames.append(frame)

    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):

    width = output_video_frames[0].shape[1]
    height = output_video_frames[0].shape[0]

    # Configura il writer per il video di output
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Usa il codec mp4v per il formato MP4
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)
    out.release()
