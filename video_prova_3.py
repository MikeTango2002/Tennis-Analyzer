from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

import cv2


def main():
    # Read Video
    input_video_path = "input_videos/video_test_3.mp4"
    video_frames,fps = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path='models/yolov8x.pt')
    player_detections = player_tracker.detect_multiple_frames(video_frames,
                                                              read_from_stub=True, #Alla prima esecuzione, impostare a False
                                                              stub_path='tracker_stubs/player_detections_video_test_3.pkl'
                                                              )
    

    ball_tracker = BallTracker(model_path='models/tennis_ball_detection_model_yolo5_best.pt')
    ball_detections = ball_tracker.detect_multiple_frames(video_frames,
                                                              read_from_stub=True, #Alla prima esecuzione, impostare a False
                                                              stub_path='tracker_stubs/ball_detections_video_test_3.pkl'
                                                              )
    
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    
    # Detect Court Lines 
    court_line_detector = CourtLineDetector(model_path='models/keypoints_model_v2.pth')
    court_keypoints = court_line_detector.predict_multiple_frames(video_frames,
                                                                    read_from_stub=False, #Alla prima esecuzione, impostare a False
                                                                    stub_path='tracker_stubs/court_keypoints_video_test_3.pkl',
                                                                    edge_detection=True,
                                                                    num_frames=30,
                                                                    window_size=30
                                                                    )
    
    # Filter Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints[0], player_detections) #Lo applico a court keypoints del primo frame, poichè i keypoints non variano da frame a frame

    # Mini Court
    mini_court = MiniCourt(video_frames[0])

    # Detect Ball Shots and Ball Bounces
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    ball_bounce_frames = ball_tracker.get_ball_bounce_frames(ball_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections, ball_detections, court_keypoints[0], ball_shot_frames) 
    
    
    # Draw output

    ## Draw Players and Ball Bounding Boxes
    
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    
    ## Draw Court Keypoints

    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    ## Draw Mini Court

    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections, color=(0, 255, 255))

    ## Draw Frame Number on top left corner

    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    save_video(output_video_frames, "output_videos/output_video_test_3.mp4", 24)
    
    


if __name__ == '__main__':
    main()