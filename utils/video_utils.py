import cv2

def read_video(video_path): #Legge il video e ritorna una lista di frame

    cap = cv2.VideoCapture(video_path) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:

        ret, frame = cap.read() #legge il frame successivo
        if not ret: 
            #non ci sono più frame da leggere
            break
        frames.append(frame) 

    cap.release() #rilascia il video
    return frames, fps

def save_video(output_video_frames, output_video_path, fps): #Salva il video

    width = output_video_frames[0].shape[1]
    height = output_video_frames[0].shape[0]
    

    # Configura il writer per il video di output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per il video di output
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) 

    for frame in output_video_frames:
        out.write(frame)
    out.release()
