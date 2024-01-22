import cv2
import numpy as np
import mysql.connector
import pandas as pd
import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QWidget, QVBoxLayout, QMessageBox
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QComboBox, QMessageBox, QLabel,QGridLayout
from PyQt5.QtGui import QIcon, QColor, QFont
import logging
import schedule
import os
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import schedule
import time
from PyQt5.QtCore import QTimer
import dlib
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt



detector = dlib.get_frontal_face_detector()
# Constants
CONFIDENCE_THRESHOLD = 0.5
BLOB_SIZE = (300, 300)
BLOB_MEAN = (104.0, 177.0, 123.0)

# Database Config
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Password01',
    'database': 'cctv_data',
    'auth_plugin': 'mysql_native_password'
}


# Model Paths
DEPLOY_PROTOTXT_PATH = "C:\\Users\\tajay\\OneDrive\\Desktop\\Semester 4\\Technical Project\\deploy.prototxt"
CAFFE_MODEL_PATH = "C:\\Users\\tajay\\OneDrive\\Desktop\\Semester 4\\Technical Project\\res10_300x300_ssd_iter_140000.caffemodel"


class MainWindow(QMainWindow):      
    def __init__(self):
        super().__init__()
        self.initUI()

        self.face_net = cv2.dnn.readNet(DEPLOY_PROTOTXT_PATH, CAFFE_MODEL_PATH)
        schedule.every().day.at("16:30").do(self.generate_report)
        self.tracked_faces_cam1 = {}
        self.tracked_faces_cam2 = {}
    
        self.setWindowTitle("Face Detection Application")
        self.setWindowIcon(QIcon('C:\\Users\\tajay\OneDrive\\Desktop\\Semester 4\\Technical Project\\icon.png'))
        self.setGeometry(100, 100, 800, 600)
        
        palette = self.palette()
        palette.setColor(self.backgroundRole(), QColor(35, 35, 35))
        self.setPalette(palette)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_scheduled_tasks)
        self.timer.start(1000)  # Check every second
        

        self.schedule_logging()

    def run_scheduled_tasks(self):
        schedule.run_pending()


    def initUI(self):
        self.setWindowTitle("Face Detection Application")
        self.setWindowIcon(QIcon('C:\\Users\\tajay\OneDrive\\Desktop\\Semester 4\\Technical Project\\icon.png'))  # Replace with your icon path
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Logo
        logo_label = QLabel()
        pixmap = QPixmap('C:\\Users\\tajay\OneDrive\\Desktop\\Semester 4\\Technical Project\\icon.png')  # Replace with your logo path
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo_label)

        # Buttons
        self.start_button = QPushButton('Start Detection')
        self.open_report_button = QPushButton('Open Previous Report')
        self.generate_report_button = QPushButton('Generate a Report')
        for button in [self.start_button, self.open_report_button, self.generate_report_button]:
            button.setFont(QFont('Arial', 12))
            layout.addWidget(button)

        # Connect button signals
        self.start_button.clicked.connect(self.start_detection)
        self.open_report_button.clicked.connect(self.open_report)
        self.generate_report_button.clicked.connect(self.generate_report)

     
        # Stylesheet
        self.setStyleSheet("""
            QPushButton {
                background-color: #4a69bd;
                color: white;
                border-radius: 10px;
                padding: 10px;
                margin: 10px;
            }
            QPushButton:hover {
                background-color: #6a89cc;
            }
        """)


        # Tooltips
        self.start_button.setToolTip("Start the face detection process")
        self.open_report_button.setToolTip("Open an existing report")
        self.generate_report_button.setToolTip("Generate a new report based on the tracked data")

        
    def open_report(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Report", "", "Excel Files (*.xlsx);;All Files (*)")
        if file_path:
            # Open the file with the default application
            try:
                if os.name == 'nt':  # For Windows
                    os.startfile(file_path)
                else:
                    opener = "open" if sys.platform == "darwin" else "xdg-open"  # For macOS and Linux
                    subprocess.call([opener, file_path])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to open the file: {e}")


      
        
    
    def calculate_and_display_average(self):
        day = self.day_combobox.currentText()
        if day:
            connection = connect_to_db(DB_CONFIG)
            cursor = connection.cursor()
            try:
                total_count, average_count = calculate_average_face_count_for_day(cursor, day)
                self.update_average_label(average_count, total_count)
            except mysql.connector.Error as err:
                logging.error(f"MySQL Error: {err}")
            finally:
                cursor.close()
                connection.close()
        else:
          logging.warning("Invalid day selected")

    def start_detection(self):
        QMessageBox.information(self, "Info", "Start Detection clicked")
        self.face_net = cv2.dnn.readNet(DEPLOY_PROTOTXT_PATH, CAFFE_MODEL_PATH)
        now = datetime.datetime.now()
        self.date_time = now.strftime("%Y-%m-%d %H:%M:%S")


        self.tracked_faces_cam1 = {}  # Initialize as an empty dictionary
        self.tracked_faces_cam2 = {}  # Initialize as an empty dictionary
        face_id_cam1 = 0
        face_id_cam2 = 0

        cap1 = cv2.VideoCapture(0)  # Use the first camera
        cap2 = cv2.VideoCapture(1)  # Use the second camera

        connection = connect_to_db(DB_CONFIG)
        cursor = connection.cursor()

        create_table(cursor)

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break
            
            detections1 = self.detect_faces_dlib(frame1)
            detections2 = self.detect_faces_dlib(frame2)

            self.tracked_faces_cam1, face_id_cam1 = self.track_faces(detections1, self.tracked_faces_cam1, face_id_cam1)
            self.tracked_faces_cam2, face_id_cam2 = self.track_faces(detections2, self.tracked_faces_cam2, face_id_cam2)

            frame1 = self.draw_faces(frame1, detections1)
            frame2 = self.draw_faces(frame2, detections2)

            cv2.putText(frame1, f'Count: {len(self.tracked_faces_cam1)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame2, f'Count: {len(self.tracked_faces_cam2)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Camera 1 Feed', frame1)
            cv2.imshow('Camera 2 Feed', frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                user_choice = QMessageBox.question(self, "Exit Application", "Do you want to exit the application?",
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if user_choice == QMessageBox.Yes:
                    break

        # Release resources and close windows
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()

        cursor.close()
        connection.close()


    def send_email(self, file_path):
        recipient_email = 'studenvisitor@gmail.com'
        sender_email = 'studenvisitor@gmail.com'  # Same as recipient in this case
        sender_password = 'lufszqynekdhzpjq'  # Replace with your Gmail password

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = 'Automated Report'

        attachment = open(file_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % file_path.split('/')[-1])

        msg.attach(part)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()



    def generate_report(self):
        
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")

        if self.tracked_faces_cam1 or self.tracked_faces_cam2:
            report_data = []

            # Process data for each camera
            total_count_cam1 = 0
            total_count_cam2 = 0
            for cam_id, tracked_faces in enumerate([self.tracked_faces_cam1, self.tracked_faces_cam2], start=1):
                for face_id, (box, timestamp) in tracked_faces.items():
                    report_data.append({
                        'Camera': f'Camera {cam_id}',
                        'Timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        'Face Count': 1,  # If you are counting faces, this might be just 1 per detection
                    })
                    if cam_id == 1:
                        total_count_cam1 += 1
                    elif cam_id == 2:
                        total_count_cam2 += 1

            df = pd.DataFrame(report_data)

            # Add total and average face count for each camera
            total_count = total_count_cam1 + total_count_cam2
            average_count = total_count / 2 if total_count > 0 else 0

            # Add a summary row to the DataFrame
            summary_data = {
                'Camera': 'Summary',
                'Camera 1 Total': total_count_cam1,
                'Camera 2 Total': total_count_cam2,
                'Total Count ': total_count,
                'Average Count': average_count
            }
            summary_df = pd.DataFrame([summary_data])

            # Combine the summary with the original DataFrame
            df = pd.concat([df, summary_df], ignore_index=True)

            # Specify the path where the report will be saved
            file_path = 'C:\\Users\\tajay\\OneDrive\\Documents\\Semester 4\\Technical Project\\report.xlsx'  # Update this to your desired path

            # Check if the directory exists, if not, create it
            directory = os.path.dirname(file_path)
            if not os.path.exists(directory):
                os.makedirs(directory)

            df.to_excel(file_path, index=False)

            # Send the report via email
            self.send_email(file_path)
        else:
            QMessageBox.information(self, "Info", "No data to generate report")

    def track_faces(self, detections, tracked_faces, face_id):
        current_time = datetime.datetime.now()

        for face in detections:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            new_face = True

            for _, (prev_box, _) in tracked_faces.items():
                # Check if there is any overlap between the new face and previously tracked faces
                # Make sure to use only the bounding box part (prev_box) for overlap checking
                overlap = not (x1 > prev_box[2] or x2 < prev_box[0] or y1 > prev_box[3] or y2 < prev_box[1])
                if overlap:
                    new_face = False
                    break

            if new_face:
                # Store the bounding box and the current time
                tracked_faces[face_id] = ((x1, y1, x2, y2), current_time)
                face_id += 1

        return tracked_faces, face_id  

    def schedule_logging(self):
        # Schedule logging to run every 5 minutes
        schedule.every(5).minutes.do(self.log_face_counts)

    def log_face_counts(self):
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        

        face_count_cam1 = len(self.tracked_faces_cam1)
        face_count_cam2 = len(self.tracked_faces_cam2)

        day_of_week = now.strftime("%A")

        connection = connect_to_db(DB_CONFIG)
        cursor = connection.cursor()

        insert_face_count(connection, cursor, face_count_cam1, day_of_week, 'Camera 1')
        insert_face_count(connection, cursor, face_count_cam2, day_of_week, 'Camera 2')

        cursor.close()
        connection.close()
    
        df = pd.DataFrame({'Camera': ['Camera 1', 'Camera 2'],
                           'Face Count': [face_count_cam1, face_count_cam2],
                           'Date Time': [date_time, date_time]})
        
        selected_time_period = self.time_period_combobox.currentText()
        write_to_excel(df, date_time, selected_time_period)
    
    def draw_faces(self, frame, detections):
        for face in detections:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
    
    def detect_faces_dlib(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        return faces



def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def connect_to_db(config):
    return mysql.connector.connect(**config)

def create_table(cursor):
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detected_faces (
        face_id INT AUTO_INCREMENT PRIMARY KEY,
        count INT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        day_of_week VARCHAR(10),
        camera_name VARCHAR(20) 
    )
    """)

def insert_face_count(connection, cursor, face_count, day_of_week, camera_name):
    cursor.execute("INSERT INTO detected_faces (count, day_of_week, camera_name) VALUES (%s, %s, %s)", (face_count, day_of_week, camera_name))
    connection.commit()

def calculate_average_face_count_for_day(cursor, day):
    cursor.execute("SELECT COUNT(*), AVG(count) FROM detected_faces WHERE day_of_week = %s", (day,))
    result = cursor.fetchone()
    total_count, average_count = result
    return total_count, average_count




    
    



def write_to_excel(df, date_time, time_period):
    df['Date Time'] = pd.to_datetime(df['Date Time'])

    # Update the period mapping to include "1 minute"
    period_mapping = {
        "1 minute": '1T',
        "5 minutes": '5T',
        "10 minutes": '10T',
        "15 minutes": '15T',
        "30 minutes": '30T',
        "1 hour": '1H'
    }

    # Resample and aggregate data
    resampled_df = df.set_index('Date Time').resample(period_mapping[time_period]).sum().reset_index()

    # Calculate average count from both cameras
    resampled_df['Average Count'] = resampled_df['Face Count'] / 2

    # Save to Excel
    file_path, _ = QFileDialog.getSaveFileName(None, "Save Excel File", "", "Excel files (*.xlsx)")
    if file_path:
        resampled_df.to_excel(file_path, index=False)
        QMessageBox.information(None, "Info", "Excel file has been created")
        
if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()