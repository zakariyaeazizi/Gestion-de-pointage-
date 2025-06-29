import cv2
import os
import face_recognition
import numpy as np
import pandas as pd
import json
from datetime import datetime
from tkinter import (
    BooleanVar, Checkbutton, Tk, Button, Label, Entry, Toplevel, Text, Scrollbar, Frame,
    VERTICAL, RIGHT, LEFT, Y, BOTH, END
)
from threading import Thread
from playsound import playsound
import time
import uuid

class FaceRecognitionSystem:
    def __init__(self):
        self.database_path = "../../frontend/public"
        self.encodings_file = "encodings.npy"
        self.names_file = "names.npy"
        self.attendance_file = "rapport_presence.csv"
        self.presence_json_file = "presence.json"
        self.person_file = "personnes.json"
        self.known_face_encodings = []
        self.known_face_names = []
        self.persons = []
        self.load_encodings()
        self.load_persons()

    def generer_absents(self):
        date_aujourdhui = datetime.now().strftime("%Y-%m-%d")

        if not os.path.exists(self.person_file):
            print("⚠️ Fichier personnes introuvable.")
            return

        with open(self.person_file, "r", encoding="utf-8") as f:
            personnes = json.load(f)

        presences = []
        if os.path.exists(self.presence_json_file):
            with open(self.presence_json_file, "r", encoding="utf-8") as f:
                toutes_presences = json.load(f)
                presences = [p for p in toutes_presences if p["date"] == date_aujourdhui]

        noms_presents = {p["nom"] for p in presences}

        absents = []
        for person in personnes:
            if person.get("active", True) and person["nom"] not in noms_presents:
                absents.append({
                    "id": person["id"],
                    "nom": person["nom"],
                    "email": person["email"],
                    "telephone": person["telephone"],
                    "image": person["image"],
                    "poste": person["poste"],
                    "departement": person["departement"],
                    "date": date_aujourdhui
                })

        with open("absent.json", "w", encoding="utf-8") as f:
            json.dump(absents, f, indent=4, ensure_ascii=False)

        print(f"✅ Fichier 'absent.json' généré avec {len(absents)} absent(s).")

    def load_encodings(self):
        try:
            if os.path.exists(self.encodings_file) and os.path.exists(self.names_file):
                encodings = np.load(self.encodings_file, allow_pickle=True)
                names = np.load(self.names_file, allow_pickle=True)
                
                if len(encodings) > 0 and isinstance(encodings[0], np.ndarray):
                    self.known_face_encodings = list(encodings)
                    self.known_face_names = list(names)
                else:
                    print("⚠️ Invalid encodings data - resetting")
                    self.known_face_encodings = []
                    self.known_face_names = []
                    self.save_encodings()
        except Exception as e:
            print(f"Error loading encodings: {e}")
            self.known_face_encodings = []
            self.known_face_names = []

    def save_encodings(self):
        try:
            encodings_array = np.array(self.known_face_encodings, dtype=np.float64)
            names_array = np.array(self.known_face_names, dtype=object)
            
            np.save(self.encodings_file, encodings_array)
            np.save(self.names_file, names_array)
        except Exception as e:
            print(f"Error saving encodings: {e}")

    def load_persons(self):
        if os.path.exists(self.person_file):
            with open(self.person_file, "r", encoding="utf-8") as f:
                self.persons = json.load(f)
        else:
            self.persons = []

    def save_persons(self):
        with open(self.person_file, "w", encoding="utf-8") as f:
            json.dump(self.persons, f, indent=4, ensure_ascii=False)

    def add_person(self, name, email, phone, poste, dep, active):
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)

        cap = cv2.VideoCapture(0)
        print("📸 Placez la personne devant la caméra. Appuyez sur 'c' pour capturer, 'q' pour annuler.")
        img_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur de lecture de la caméra")
                break

            cv2.imshow("Capture - Appuyez sur 'c'", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                img_path = os.path.join(self.database_path, filename)
                cv2.imwrite(img_path, frame)
                print(f"✅ {name} ajouté avec succès ! Image sauvegardée à {img_path}")
                break
            elif key == ord('q'):
                print("Ajout annulé.")
                break

        cap.release()
        cv2.destroyAllWindows()

        if img_path:
            image = face_recognition.load_image_file(img_path)
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                print("❌ Aucun visage détecté.")
                return
            face_encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            self.save_encodings()

            person_id = str(uuid.uuid4())
            now = datetime.now().isoformat()
            self.persons.append({
                "id": person_id,
                "nom": name,
                "email": email,
                "telephone": phone,
                "poste": poste,
                "departement": dep,
                "image": img_path,
                "date_creation": now,
                "date_modification": now,
                "active": active 
            })
            self.save_persons()

    def supprimer_personne(self, name_to_delete):
        if name_to_delete in self.known_face_names:
            index = self.known_face_names.index(name_to_delete)
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
            self.save_encodings()
            self.persons = [p for p in self.persons if p["nom"] != name_to_delete]
            self.save_persons()

            for f in os.listdir(self.database_path):
                if f.startswith(name_to_delete + "_"):
                    try:
                        os.remove(os.path.join(self.database_path, f))
                    except Exception as e:
                        print(f"Erreur suppression fichier {f}: {e}")
            return True
        else:
            return False

    def save_presence_json(self, name, date_today, time_now):
        if os.path.exists(self.presence_json_file):
            with open(self.presence_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        if not any(p["nom"] == name and p["date"] == date_today for p in data):
            image_path = ""
            person_id = ""
            for person in self.persons:
                if person["nom"] == name:
                    image_path = person.get("image", "")
                    person_id = person.get("id", "")
                    break
            data.append({
                "nom": name,
                "date": date_today,
                "heure": time_now,
                "image": image_path,
                "person_id": person_id
            })
            with open(self.presence_json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

    def mark_attendance(self, name):
        date_today = datetime.now().strftime("%Y-%m-%d")
        time_now = datetime.now().strftime("%H:%M:%S")

        if not os.path.exists(self.attendance_file):
            df = pd.DataFrame(columns=["Nom", "Date", "Heure"])
        else:
            df = pd.read_csv(self.attendance_file)

        if not ((df['Nom'] == name) & (df['Date'] == date_today)).any():
            new_row = {"Nom": name, "Date": date_today, "Heure": time_now}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.attendance_file, index=False)

        self.save_presence_json(name, date_today, time_now)

    def start_recognition(self):
        self.load_encodings()
        print("🎥 Démarrage de la reconnaissance. Appuyez sur 'q' pour quitter.")

        if not self.known_face_encodings:
            print("⚠️ Base vide.")
            return

        video_capture = cv2.VideoCapture(0)
        pause_until = None
        validated_name = ""
        paused_frame = None

        while True:
            if pause_until:
                if datetime.now() < pause_until:
                    frame = paused_frame.copy()
                    message = f" Validation : {validated_name}"
                    (text_width, text_height), baseline = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
                    x, y = 30, 50
                    cv2.rectangle(frame, (x-5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), cv2.FILLED)
                    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Reconnaissance faciale', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                else:
                    pause_until = None
                    paused_frame = None

            ret, frame = video_capture.read()
            if not ret:
                print("❌ Erreur caméra.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)

            if not face_locations:
                cv2.imshow('Reconnaissance faciale', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            face_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)

            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = []
                face_distances = []
                name = "Inconnu"
                
                try:
                    if not isinstance(face_encoding, np.ndarray) or face_encoding.shape != (128,):
                        print("⚠️ Encodage facial invalide")
                        continue
                    
                    if len(self.known_face_encodings) == 0:
                        print("⚠️ Aucun visage connu chargé")
                        continue

                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding,
                        tolerance=0.6
                    )
                    
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )

                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            self.mark_attendance(name)
                            validated_name = name
                            Thread(target=playsound, args=("success.mp3",), daemon=True).start()
                            pause_until = datetime.now() + pd.Timedelta(seconds=10).to_pytimedelta()
                            paused_frame = frame.copy()

                except Exception as e:
                    print(f"❌ Erreur de reconnaissance: {str(e)}")
                    continue

                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Reconnaissance faciale', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Système Reconnaissance Faciale RH")
        self.root.geometry("600x450")
        self.fr_system = FaceRecognitionSystem()
        self.schedule_absents_generation()

        button_frame = Frame(root)
        button_frame.pack(pady=20)

        buttons = [
            ("Ajouter une personne", self.add_person_window),
            ("Supprimer une personne", self.delete_person_window),
            ("Démarrer la reconnaissance", lambda: Thread(target=self.fr_system.start_recognition, daemon=True).start()),
            ("Afficher présences", self.show_attendance_window),
            ("Lister les personnes", self.show_person_list_window),
            ("Générer liste des absents", self.generer_absents),
            ("Quitter", root.quit)
        ]

        for text, command in buttons:
            Button(button_frame, text=text, width=30, height=2, command=command).pack(pady=5)

    def schedule_absents_generation(self):
        self.fr_system.generer_absents()
        self.root.after(30000, self.schedule_absents_generation)
    
    def add_person_window(self):
        win = Toplevel(self.root)
        win.title("Ajouter une personne")

        Label(win, text="Nom complet").pack()
        entry_name = Entry(win)
        entry_name.pack()

        Label(win, text="Email").pack()
        entry_email = Entry(win)
        entry_email.pack()

        Label(win, text="Téléphone").pack()
        entry_phone = Entry(win)
        entry_phone.pack()

        Label(win, text="poste").pack()
        entry_poste = Entry(win)
        entry_poste.pack()

        Label(win, text="departement").pack()
        entry_dep = Entry(win)
        entry_dep.pack()

        active_var = BooleanVar(value=True)
        Checkbutton(win, text="Actif", variable=active_var).pack()

        def on_add():
            name = entry_name.get().strip()
            email = entry_email.get().strip()
            phone = entry_phone.get().strip()
            poste = entry_poste.get().strip()
            departement = entry_dep.get().strip()
            active = active_var.get()
            if not name or not email or not phone:
                print("Tous les champs sont obligatoires")
                return
            win.destroy()
            self.fr_system.add_person(name, email, phone, poste, departement, active)

        Button(win, text="Ajouter", command=on_add).pack(pady=10)

    def delete_person_window(self):
        win = Toplevel(self.root)
        win.title("Supprimer une personne")

        Label(win, text="Nom complet à supprimer").pack()
        entry_name = Entry(win)
        entry_name.pack()

        def on_delete():
            name = entry_name.get().strip()
            if not name:
                print("Entrez un nom")
                return
            success = self.fr_system.supprimer_personne(name)
            print(f"{name} supprimé." if success else f"{name} non trouvé.")
            win.destroy()

        Button(win, text="Supprimer", command=on_delete).pack(pady=10)

    def show_attendance_window(self):
        win = Toplevel(self.root)
        win.title("Présences enregistrées")
        win.geometry("500x400")

        text = Text(win)
        text.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(win, orient=VERTICAL, command=text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        text.config(yscrollcommand=scrollbar.set)

        if os.path.exists(self.fr_system.presence_json_file):
            with open(self.fr_system.presence_json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for presence in data:
                text.insert(END, f"Nom: {presence['nom']} | Date: {presence['date']} | Heure: {presence['heure']}\n")
        else:
            text.insert(END, "Aucune présence enregistrée.")

    def show_person_list_window(self):
        win = Toplevel(self.root)
        win.title("Liste des personnes enregistrées")
        win.geometry("500x400")

        text = Text(win)
        text.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = Scrollbar(win, orient=VERTICAL, command=text.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        text.config(yscrollcommand=scrollbar.set)

        if os.path.exists(self.fr_system.person_file):
            with open(self.fr_system.person_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for person in data:
                text.insert(END, f"ID: {person['id']}\nNom: {person['nom']}\nEmail: {person['email']}\nTéléphone: {person['telephone']}\nActif: {'Oui' if person.get('active') else 'Non'}\n\n")
        else:
            text.insert(END, "Aucune personne enregistrée.")

    def generer_absents(self):
        self.fr_system.generer_absents()

if __name__ == "__main__":
    if not os.path.exists("success.mp3"):
        print("⚠️ Veuillez ajouter un fichier 'success.mp3' dans le dossier.")
    root = Tk()
    app = App(root)
    root.mainloop()