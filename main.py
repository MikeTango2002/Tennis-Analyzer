from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector


def main():
    # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    #save_video(video_frames, "output_videos/prova_output.mp4")
    #return

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    player_detections = player_tracker.detect_multiple_frames(video_frames,
                                                              read_from_stub=True , #Alla prima esecuzione, impostare a False
                                                              stub_path='tracker_stubs/player_detections.pkl'
                                                              )
    

    ball_tracker = BallTracker(model_path='models/tennis_ball_detection_model_yolo5_last.pt')
    ball_detections = ball_tracker.detect_multiple_frames(video_frames,
                                                              read_from_stub=True, 
                                                              stub_path='tracker_stubs/ball_detections.pkl'
                                                              )
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Detect Court Lines
    court_line_detector = CourtLineDetector(model_path='models/keypoints_model_v2.pth')
    court_keypoints = court_line_detector.predict_multiple_frames(video_frames,
                                                                    read_from_stub=True, 
                                                                    stub_path='tracker_stubs/court_keypoints.pkl',
                                                                    edge_detection=True,
                                                                    num_frames=5,
                                                                    window_size=30
                                                                    )
    

    # Draw output

    ## Draw Players and Ball Bounding Boxes
    
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    ## Draw Court Keypoints

    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)


    save_video(output_video_frames, "output_videos/output_video.mp4")
    
    


if __name__ == '__main__':
    main()