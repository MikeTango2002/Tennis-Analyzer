from ultralytics import YOLO
import cv2

class PlayerTracker:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_multiple_frames(self, frames): #Ritorna la lista delle detections di ciascun frame

        player_detections = []

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        return player_detections

    def detect_frame(self, frame): #Ritorna la lista delle detection di un singolo frame

        results = self.track(frame, persist=True)[0] 
        #Se il parametro persist è impostato su True, la funzione manterrà informazioni sugli oggetti tracciati da frame precedenti, permettendo così di seguire oggetti in movimento attraverso più frame.
        #Bisogna inoltre estrarre solo le informazioni relative alle persone, che hanno box id = 0

        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0] #Per ottenere le coordinate dei bounding box in formato xmin ymin, xmax, ymax
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id] #Per associare l'id al nome

            if object_cls_name == 'person':
                player_dict[track_id] = result
        
        return player_dict
    
    def draw_bboxes(self, video_frames, player_detections):

        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
        #La funzione zip() di Python prende più iterabili (come liste, tuple o altri oggetti iterabili) e 
        #combina i loro elementi in coppie, terne, ecc., restituendo un iteratore di tuple. 
        #Ogni tupla contiene elementi provenienti dalle diverse iterabili in posizione corrispondente.

            #Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) #Il 2 indica che sarà colorato solo il bordo
            output_video_frames.append(frame)

        return output_video_frames


