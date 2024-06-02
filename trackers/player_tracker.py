from ultralytics import YOLO
import cv2
import pickle 
import sys
sys.path.append('../') #Per poter vedere i moduli presenti in utils
from utils import get_center_of_bbox, measure_distance

class PlayerTracker: #Classe per il tracciamento dei giocatori

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections): 
        #Ritorna le detections dei giocatori, che sono le persone che si trovano all'interno del campo

        player_detections_first_frame = player_detections[0] #Prende le detections del primo frame
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame) 
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = [] #Lista delle distanze tra i giocatori e i keypoints del campo (tuple del tipo: track_id, min_distance)
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox) #Calcola il centro del bounding box del giocatore
            
            #Calcola la distanza tra il centro del bounding box del giocatore e i keypoints del campo
            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        #Ordina le distanze in ordine crescente
        distances.sort(key=lambda x: x[1]) 

        #Prendi i primi 2 track_id, che sono i giocatori

        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players
        

    def detect_multiple_frames(self, frames, read_from_stub=False, stub_path=None): 
        #Ritorna la lista delle detections di ciascun frame

        player_detections = [] 

        if read_from_stub and stub_path is not None: 
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames: #Per ogni frame, rileva i giocatori e salva le detections in una lista
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None: #Se è stato specificato un percorso per il file stub, salva le detections in un file stub
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame): #Ritorna la lista delle detection di un singolo frame

        results = self.model.track(frame, persist=True)[0] 
        #La funzione track() prende in input un frame e restituisce una lista di oggetti rilevati in quel frame. L'indice 0 è utilizzato per ottenere le informazioni sui box, le classi e gli id degli oggetti rilevati.
        #Se il parametro persist è impostato su True, la funzione manterrà informazioni sugli oggetti tracciati da frame precedenti, permettendo così di seguire oggetti in movimento attraverso più frame.
        #Bisogna inoltre estrarre solo le informazioni relative alle persone, che hanno box_cls_id = 0

        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue
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
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) #Il 2 indica che sarà colorato solo il bordo
            output_video_frames.append(frame)

        return output_video_frames


