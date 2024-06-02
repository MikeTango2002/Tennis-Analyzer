from ultralytics import YOLO
import cv2
import pickle 
import pandas as pd

class BallTracker: #Classe per il tracciamento della pallina

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions] #Prende le posizioni della pallina da ogni frame e in caso di non rilevamento, restituisce una lista vuota
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2']) #Crea un DataFrame con le posizioni della pallina

        df_ball_positions = df_ball_positions.interpolate(method='linear') #Interpolazione lineare per riempire i valori mancanti
        df_ball_positions = df_ball_positions.bfill() #Riempe un eventuale valore mancante all'inizio del dataframe con i valori successivi

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()] #Converte il DataFrame in un array numpy e crea un dizionario con le posizioni della pallina per ogni frame

        return ball_positions
    
    
    def detect_multiple_frames(self, frames, read_from_stub=False, stub_path=None): #Ritorna la lista delle detections di ciascun frame

        ball_detections = [] 

        if read_from_stub and stub_path is not None: 
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames: #Per ogni frame, rileva i giocatori e salva le detections in una lista
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None: #Se è stato specificato un percorso per il file stub, salva le detections in un file stub
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame): #Ritorna la lista delle detection di un singolo frame

        results = self.model.predict(frame, conf=0.15)[0] 
        #La funzione predict() prende in input un frame e restituisce una lista di oggetti rilevati in quel frame. L'indice 0 è utilizzato per ottenere le informazioni sui box, le classi e gli id degli oggetti rilevati.
        
        ball_dict = {}
        for box in results.boxes:
          
            result = box.xyxy.tolist()[0] #Per ottenere le coordinate dei bounding box in formato xmin ymin, xmax, ymax

            ball_dict[1] = result #Salva le coordinate del bounding box della pallina (essendoci solo la classe della pallina nel dizionario, l'id è 1)
        
        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections):

        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
        #La funzione zip() di Python prende più iterabili (come liste, tuple o altri oggetti iterabili) e 
        #combina i loro elementi in coppie, terne, ecc., restituendo un iteratore di tuple. 
        #Ogni tupla contiene elementi provenienti dalle diverse iterabili in posizione corrispondente.

            #Draw Bounding Boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2) #Il 2 indica che sarà colorato solo il bordo
            output_video_frames.append(frame)

        return output_video_frames


