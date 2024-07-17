from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivy.animation import Animation

import cv2
import face_recognition
import os
import csv
import numpy as np
import pyttsx3
import sounddevice as sd
import wave
import speech_recognition as sr
import wikipedia
import threading

# Initialize pyttsx3 engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # Speed of speech

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def record_audio():
    audio_data = sd.rec(int(4 * 44100), samplerate=44100, channels=2, dtype='int16')
    sd.wait()
    audio_data = audio_data.tobytes()

    with wave.open('recorded_audio.wav', 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(audio_data)

def recognize_speech():
    r = sr.Recognizer()
    with sr.AudioFile('recorded_audio.wav') as source:
        try:
            audio = r.record(source)
            audio_text = r.recognize_google(audio, language='en-in')
            return audio_text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""

def load_known_faces():
    known_faces = []
    try:
        with open('known_faces.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                name = row[0]
                encoding = np.array([float(val) for val in row[1:]])
                known_faces.append((name, encoding))
    except FileNotFoundError:
        pass
    return known_faces

def save_known_faces(known_faces):
    with open('known_faces.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for name, encoding in known_faces:
            encoding_as_strings = [str(val) for val in encoding]
            row = [name] + encoding_as_strings
            writer.writerow(row)

def search_wikipedia(user_query):
    try:
        result = wikipedia.summary(user_query, sentences=3)
        return result
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple results found for {user_query}. Please specify."
    except wikipedia.exceptions.PageError as e:
        return f"Sorry, {user_query} page does not exist."

class MyApp(App):
    def build(self):
        layout = FloatLayout()

        # Load and display the GIF image to fill the entire layout
        gif_image = AsyncImage(source='askme.jpg', size_hint=(1, 1), pos_hint={'top': 1})
        layout.add_widget(gif_image)

        self.query_label = Label(
            text="Query: ",
            size_hint=(0.8, 0.05),
            pos_hint={'x': 0.1, 'y': 0.7},
            color=(1, 1, 1, 1),  # White text color
            font_size='18sp'
        )
        layout.add_widget(self.query_label)

        self.response_label = Label(
            text="Response: ",
            size_hint=(0.8, 0.05),
            pos_hint={'x': 0.1, 'y': 0.65},
            color=(1, 1, 1, 1),  # White text color
            font_size='18sp'
        )
        layout.add_widget(self.response_label)

        self.button = Button(
            text='Start AI',
            size_hint=(0.5, 0.1),
            pos_hint={'center_x': 0.5, 'y': 0},
            background_normal='',  # Clear default background
            background_color=(139/255, 0, 0, 1),  # Red background
            color=(0, 1, 0, 1),  # Green text color
            font_size='20sp',
            bold=True,
            border=(10, 20, 20, 10),  # Adjusted border radius
            background_down='down_button.jpg'  # Image when button is pressed
        )
        self.button.bind(on_press=self.start_ai)
        layout.add_widget(self.button)

        Window.clearcolor = (0.2, 0.2, 0.2, 1)  # Set window background to dark gray

        return layout

    def start_ai(self, instance):
        threading.Thread(target=self.main_process).start()

    def main_process(self):
        video_capture = cv2.VideoCapture(0)
        known_faces = load_known_faces()

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                name = "Unknown"
                match_found = False

                for known_face in known_faces:
                    known_encoding = known_face[1]
                    if len(known_encoding) == 128:
                        distance = face_recognition.face_distance([known_encoding], face_encoding)
                        if distance[0] < 0.6:
                            name = known_face[0]
                            match_found = True
                            break

                if not match_found:
                    speak("I didn't see you before. Could you please tell me, what is your name?")
                    record_audio()
                    user_name = recognize_speech()
                    if user_name:
                        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                        known_faces.append((user_name, face_encoding))
                        save_known_faces(known_faces)
                        filename = f"{user_name}.jpg"
                        cv2.imwrite(filename, frame)
                        speak(f"{user_name}, it's an honor to meet you.")
                if match_found:
                    speak(f"Do you have any doubts, {name}?")
                    record_audio()
                    user_query = recognize_speech()
                    self.query_label.text = f"Query: {user_query}"
                   
                    speak(f"Do you mean {user_query}?")
                    record_audio()
                    user_response = recognize_speech().lower()
                    self.response_label.text = f"Response: {user_response}"
                    
                    if user_response in ['no', 'no i dont', 'i dont']:   
                        speak("Sorry, could you please repeat?")
                        record_audio()
                        user_query = recognize_speech()
                        self.query_label.text = f"Query: {user_query}"
                    elif user_response in ['yes', 'yes i do', 'i do']:
                        response = search_wikipedia(user_query)
                        speak(response)
                        self.response_label.text = f"Response: {response}"
                        print(f"Response: {response}")
                    else:
                        print(f"Response: {user_response}")

                cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                break

            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

    def show_popup(self, title, message):
        popup_layout = BoxLayout(orientation='vertical', padding=10)
        popup_label = Label(text=message)
        close_button = Button(text='Close', size_hint=(1, 0.2))
        popup_layout.add_widget(popup_label)
        popup_layout.add_widget(close_button)
        popup = Popup(title=title, content=popup_layout, size_hint=(0.6, 0.4))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    MyApp().run()
