import datetime
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import os

from google.cloud import storage
from google.oauth2 import service_account
import cv2
import face_recognition
import os
import numpy as np
import pickle
import datetime
def photos_print(video_path):
  count = 0
  cap = cv2.VideoCapture(video_path)

  frame_counter = 0
  attendance_dict = {}  # Dictionary to store attendance data

  # Get the original frame size
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Calculate the cropping coordinates
  crop_x = (width - min(width, height)) // 2
  crop_y = (height - min(width, height)) // 2
  crop_width = min(width, height)
  crop_height = min(width, height)

  # Desired square frame size
  square_size = 500

  # Output video writer
  #out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (square_size, square_size))

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break
      count += 1

      if count % 20 !=0:
        continue

      cropped_frame = frame[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

      # Resize the square portion to the desired square frame size
      frame = cv2.resize(cropped_frame, (square_size, square_size))

      # Find faces in the frame
      face_locations = face_recognition.face_locations(frame)
      face_encodings = face_recognition.face_encodings(frame, face_locations)

      if len(face_locations) == 0:
          # Skip the frame if no faces are detected
          continue

      # Iterate over each detected face
      for face_encoding, face_location in zip(face_encodings, face_locations):
          # Compare face encoding with the known faces
          matches = face_recognition.compare_faces(known_faces, face_encoding)
          name = "Unknown"

          # Find the best match
          if len(matches) > 0:
              face_distances = face_recognition.face_distance(known_faces, face_encoding)
              best_match_index = np.argmin(face_distances)
              if matches[best_match_index]:
                  name = known_names[best_match_index]

                  # Update attendance dictionary with name and timestamp
                  attendance_dict[name] = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")

              # Draw a box around the face and label the name
              top, right, bottom, left = face_location
              cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
              cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

      # Write the frame to the output video
      #out.write(frame)

      # Display the resulting frame
      cv2_imshow(frame)
      print(count)

  cap.release()
  #out.release()
  cv2.destroyAllWindows()
  return attendance_dict


def convert_into_csv(a):
# Convert attendance dictionary to DataFrame
    df = pd.DataFrame.from_dict(a, orient='index', columns=['Timestamp'])

    # Split timestamp into separate date and time columns
    df[['Date', 'Time']] = df['Timestamp'].str.split(' ', 1, expand=True)

    # Rename the first column as "Name"
    df = df.rename(columns={0: 'Name'})

    df.to_csv('Attendance.csv',index = False)     