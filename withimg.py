import cv2
import dlib
import time
import datetime
from scipy.spatial import distance
import smtplib
from playsound import playsound
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.text import MIMEText


# Initialize the face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the threshold for closed eyes
EYE_AR_THRESH = 0.25

# Initialize variables
COUNTER = 0
ALERTED = False
EMAIL_SENT=False
FIRST_WARNING = False

# Define the function to calculate the eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def send_mail():
    # Enter your email details

    sender = "workanalyzer321@gmail.com"
    receiver = "jaison.inmca2025@saintgits.org"
    subject = 'Test Email with Image Attachment.'

    # create a message object
    message = MIMEMultipart()
    message['From'] = sender
    message['To'] = receiver
    message['Subject'] = subject

    with open('capture.jpg', 'rb') as file:
         image_data = file.read()
         image = MIMEImage(image_data, name='capture.jpg')
         message.attach(image)

          # add some text to the email body
    body = "The person is showing signs of drowsiness."
    message.attach(MIMEText(body))


     # create an SMTP session to send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.starttls()
        smtp.login(sender, 'uobebtfigeaogwip')
        smtp.sendmail(sender, receiver, message.as_string())


   
    print("Email sent successfully!")

    #set EMAIL_SENT to TRUE
    global EMAIL_SENT
    EMAIL_SENT = True


while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray, 0)

    # Loop over the faces detected
    for face in faces:
        # Get the facial landmarks for the face
        landmarks = predictor(gray, face)

        # Extract the left and right eye coordinates
        left_eye = []
        right_eye = []
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Calculate the EAR for each eye
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Average the EAR for both eyes
        ear = (left_ear + right_ear) / 2.0

        # Check if the EAR is below the threshold
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # If the eyes have been closed for a certain number of frames, send an alert
            if COUNTER == 15 and not ALERTED:
                if not FIRST_WARNING:
                    playsound('woan.mp3')
                    print("First Warning!")
                    FIRST_WARNING = True
                else:
                    if not EMAIL_SENT:
                        cv2.imwrite('capture.jpg', frame)
                        send_mail()
                        EMAIL_SENT = True
                        ALERTED = True
                        print("Drowsiness Alert!")
                        break
        else:
            COUNTER = 0
            ALERTED = False

    # Display the video feed
    cv2.imshow("Video Feed", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
