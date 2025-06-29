from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import uuid
from datetime import datetime
import base64
import cv2
import numpy as np
import face_recognition

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})


# Configuration
UPLOAD_FOLDER = 'uploads/images'
DATA_FOLDER = 'data'
PERSONS_FILE = os.path.join(DATA_FOLDER, 'personnes.json')
PRESENCE_FILE = os.path.join(DATA_FOLDER, 'presence.json')
ABSENT_FILE = os.path.join(DATA_FOLDER, 'absent.json')
ENCODINGS_FILE = os.path.join(DATA_FOLDER, 'encodings.npy')  # Fichier pour les encodages faciaux
NAMES_FILE = os.path.join(DATA_FOLDER, 'names.npy')         # Fichier pour les noms associés

# Créer les dossiers nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# Initialiser les fichiers de reconnaissance faciale s'ils n'existent pas
if not os.path.exists(ENCODINGS_FILE):
    np.save(ENCODINGS_FILE, np.array([], dtype=object))
if not os.path.exists(NAMES_FILE):
    np.save(NAMES_FILE, np.array([], dtype=object))


# Créer les dossiers nécessaires
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

def load_json_file(filename, default=[]):
    """Charger un fichier JSON"""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erreur lors du chargement de {filename}: {e}")
    return default

def save_json_file(filename, data):
    """Sauvegarder un fichier JSON"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de {filename}: {e}")
        return False

# Routes pour les personnes
@app.route('/api/persons', methods=['GET'])
def get_persons():
    """Récupérer toutes les personnes"""
    try:
        persons = load_json_file(PERSONS_FILE)
        return jsonify({
            'success': True,
            'data': persons,
            'total': len(persons)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/persons/<person_id>', methods=['GET'])
def get_person(person_id):
    """Récupérer une personne par ID"""
    try:
        persons = load_json_file(PERSONS_FILE)
        person = next((p for p in persons if p['id'] == person_id), None)
        
        if person:
            return jsonify({'success': True, 'data': person})
        else:
            return jsonify({'success': False, 'message': 'Personne non trouvée'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/persons', methods=['POST'])
def add_person():
    """Ajouter une nouvelle personne avec association du visage"""
    try:
        data = request.get_json()
        
        # Validation des données
        required_fields = ['nom', 'email', 'telephone', 'image_filename']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False, 
                    'message': f'Le champ {field} est requis'
                }), 400
        
        persons = load_json_file(PERSONS_FILE)
        
        # Vérifier si l'email existe déjà
        if any(p['email'] == data['email'] for p in persons):
            return jsonify({
                'success': False, 
                'message': 'Cette adresse email existe déjà'
            }), 400
        
        # Charger les encodages existants
        known_face_names = np.load(NAMES_FILE, allow_pickle=True).tolist()
        
        # Associer le nom au dernier encodage ajouté
        known_face_names.append(data['nom'])
        np.save(NAMES_FILE, np.array(known_face_names, dtype=object))
        
        # Créer une nouvelle personne
        new_person = {
            'id': str(uuid.uuid4()),
            'nom': data['nom'],
            'email': data['email'],
            'telephone': data['telephone'],
            'poste': data.get('poste', ''),
            'departement': data.get('departement', ''),
            'date_creation': datetime.now().isoformat(),
            'image': data['image_filename'],
            'active': True
        }
        
        persons.append(new_person)
        
        if save_json_file(PERSONS_FILE, persons):
            return jsonify({
                'success': True, 
                'message': 'Personne ajoutée avec succès',
                'data': new_person
            }), 201
        else:
            return jsonify({
                'success': False, 
                'message': 'Erreur lors de la sauvegarde'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/api/persons/<person_id>', methods=['PUT'])
def update_person(person_id):
    """Modifier une personne"""
    try:
        data = request.get_json()
        persons = load_json_file(PERSONS_FILE)
        
        person_index = next((i for i, p in enumerate(persons) if p['id'] == person_id), None)
        
        if person_index is None:
            return jsonify({'success': False, 'message': 'Personne non trouvée'}), 404
        
        # Mettre à jour les champs
        updatable_fields = ['nom', 'email', 'telephone', 'poste', 'departement', 'active']
        for field in updatable_fields:
            if field in data:
                persons[person_index][field] = data[field]
        
        persons[person_index]['date_modification'] = datetime.now().isoformat()
        
        if save_json_file(PERSONS_FILE, persons):
            return jsonify({
                'success': True, 
                'message': 'Personne modifiée avec succès',
                'data': persons[person_index]
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Erreur lors de la sauvegarde'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/persons/<person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Supprimer une personne"""
    try:
        persons = load_json_file(PERSONS_FILE)
        person = next((p for p in persons if p['id'] == person_id), None)
        
        if not person:
            return jsonify({'success': False, 'message': 'Personne non trouvée'}), 404
        
        # Supprimer la personne
        persons = [p for p in persons if p['id'] != person_id]
        
        if save_json_file(PERSONS_FILE, persons):
            return jsonify({
                'success': True, 
                'message': 'Personne supprimée avec succès'
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Erreur lors de la suppression'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    
@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """Reconnaître un visage à partir d'une image"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Aucun fichier sélectionné'}), 400
        
        # Sauvegarde temporaire
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        file.save(temp_path)
        
        # Charger l'image
        unknown_image = face_recognition.load_image_file(temp_path)
        face_locations = face_recognition.face_locations(unknown_image)
        
        if not face_locations:
            os.remove(temp_path)
            return jsonify({
                'success': False,
                'message': 'Aucun visage détecté dans l\'image'
            }), 400
        
        # Encodage du visage inconnu
        unknown_encoding = face_recognition.face_encodings(unknown_image, [face_locations[0]])[0]
        
        # Charger les encodages connus
        known_face_encodings = np.load(ENCODINGS_FILE, allow_pickle=True).tolist()
        known_face_names = np.load(NAMES_FILE, allow_pickle=True).tolist()
        
        # Comparer avec les visages connus
        matches = face_recognition.compare_faces(known_face_encodings, unknown_encoding)
        name = "Inconnu"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # Nettoyer le fichier temporaire
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'recognized': name != "Inconnu",
            'name': name,
            'message': 'Reconnaissance terminée'
        })
        
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/encode-all', methods=['POST'])
def encode_all_faces():
    """Encoder tous les visages des personnes existantes"""
    try:
        persons = load_json_file(PERSONS_FILE)
        known_face_encodings = []
        known_face_names = []
        
        for person in persons:
            if person.get('image'):
                try:
                    image_path = os.path.join(UPLOAD_FOLDER, os.path.basename(person['image']))
                    if os.path.exists(image_path):
                        image = face_recognition.load_image_file(image_path)
                        face_locations = face_recognition.face_locations(image)
                        
                        if face_locations:
                            encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                            known_face_encodings.append(encoding)
                            known_face_names.append(person['nom'])
                except Exception as e:
                    print(f"Erreur avec {person['nom']}: {str(e)}")
                    continue
        
        # Sauvegarder les encodages
        np.save(ENCODINGS_FILE, np.array(known_face_encodings, dtype=object))
        np.save(NAMES_FILE, np.array(known_face_names, dtype=object))
        
        return jsonify({
            'success': True,
            'message': f'{len(known_face_names)} visages encodés avec succès'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Routes pour les présences
@app.route('/api/presences', methods=['GET'])
def get_presences():
    """Récupérer toutes les présences"""
    try:
        presences = load_json_file(PRESENCE_FILE)
        
        # Filtres optionnels
        person_id = request.args.get('person_id')
        date_from = request.args.get('date_from')
        date_to = request.args.get('date_to')
        
        filtered_presences = presences
        
        if person_id:
            filtered_presences = [p for p in filtered_presences if p.get('person_id') == person_id]
        
        if date_from:
            filtered_presences = [p for p in filtered_presences if p.get('date') >= date_from]
        
        if date_to:
            filtered_presences = [p for p in filtered_presences if p.get('date') <= date_to]
        
        return jsonify({
            'success': True,
            'data': filtered_presences,
            'total': len(filtered_presences)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    
# Routes pour les présences


@app.route('/api/presences/person/<person_id>', methods=['GET'])
def get_person_presences(person_id):
    """Récupérer les présences d'une personne"""
    try:
        presences = load_json_file(PRESENCE_FILE)
        persons = load_json_file(PERSONS_FILE)
        
        # Trouver la personne
        person = next((p for p in persons if p['id'] == person_id), None)
        if not person:
            return jsonify({'success': False, 'message': 'Personne non trouvée'}), 404
        
        # Filtrer les présences pour cette personne
        person_presences = [p for p in presences if p.get('person_id') == person_id]
        
        # Trier par date décroissante
        person_presences.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'person': person,
            'presences': person_presences,
            'total': len(person_presences)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/presences', methods=['POST'])
def add_presence():
    """Ajouter une présence (utilisé par le système de reconnaissance)"""
    try:
        data = request.get_json()
        
        required_fields = ['person_id', 'nom']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False, 
                    'message': f'Le champ {field} est requis'
                }), 400
        
        presences = load_json_file(PRESENCE_FILE)
        
        # Créer une nouvelle présence
        new_presence = {
            'id': str(uuid.uuid4()),
            'person_id': data['person_id'],
            'nom': data['nom'],
            'date': data.get('date', datetime.now().strftime('%Y-%m-%d')),
            'heure': data.get('heure', datetime.now().strftime('%H:%M:%S')),
            'timestamp': datetime.now().isoformat()
        }
        
        # Vérifier si la personne est déjà présente aujourd'hui
        today = new_presence['date']
        existing_presence = next((p for p in presences 
                                if p['person_id'] == data['person_id'] 
                                and p['date'] == today), None)
        
        if existing_presence:
            return jsonify({
                'success': False, 
                'message': 'Présence déjà enregistrée pour aujourd\'hui'
            }), 400
        
        presences.append(new_presence)
        
        if save_json_file(PRESENCE_FILE, presences):
            return jsonify({
                'success': True, 
                'message': 'Présence enregistrée avec succès',
                'data': new_presence
            }), 201
        else:
            return jsonify({
                'success': False, 
                'message': 'Erreur lors de la sauvegarde'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Routes pour les recherches
@app.route('/api/search/persons', methods=['GET'])
def search_persons():
    """Rechercher des personnes"""
    try:
        query = request.args.get('q', '').lower()
        persons = load_json_file(PERSONS_FILE)
        
        if not query:
            return jsonify({'success': True, 'data': persons})
        
        # Recherche dans nom, email, poste, département
        filtered_persons = []
        for person in persons:
            if (query in person.get('nom', '').lower() or 
                query in person.get('email', '').lower() or 
                query in person.get('poste', '').lower() or 
                query in person.get('departement', '').lower()):
                filtered_persons.append(person)
        
        return jsonify({
            'success': True,
            'data': filtered_persons,
            'total': len(filtered_persons)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Route pour les statistiques
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Récupérer les statistiques du système"""
    try:
        persons = load_json_file(PERSONS_FILE)
        presences = load_json_file(PRESENCE_FILE)
        
        # Statistiques de base
        total_persons = len(persons)
        active_persons = len([p for p in persons if p.get('active', True)])
        
        # Présences d'aujourd'hui
        today = datetime.now().strftime('%Y-%m-%d')
        today_presences = len([p for p in presences if p.get('date') == today])
        
        # Présences de cette semaine
        from datetime import timedelta
        week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        week_presences = len([p for p in presences if p.get('date') >= week_ago])
        
        return jsonify({
            'success': True,
            'data': {
                'total_persons': total_persons,
                'active_persons': active_persons,
                'today_presences': today_presences,
                'week_presences': week_presences,
                'total_presences': len(presences)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Route pour uploader des images
@app.route('/api/upload/image', methods=['POST'])
def upload_image():
    """Upload d'image et encodage du visage"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'message': 'Aucune image fournie'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Aucun fichier sélectionné'}), 400
        
        # Vérification de l'extension
        if '.' not in file.filename:
            return jsonify({'success': False, 'message': 'Extension de fichier manquante'}), 400
            
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        if file_extension not in ['jpg', 'jpeg', 'png']:
            return jsonify({'success': False, 'message': 'Format d\'image non supporté'}), 400
        
        # Génération d'un nom de fichier unique
        filename = f"{uuid.uuid4()}.{file_extension}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Sauvegarde temporaire pour le traitement
        file.save(filepath)
        
        # Chargement et encodage du visage
        image = face_recognition.load_image_file(filepath)
        face_locations = face_recognition.face_locations(image)
        
        if not face_locations:
            os.remove(filepath)
            return jsonify({
                'success': False,
                'message': 'Aucun visage détecté dans l\'image'
            }), 400
        
        # Encodage du premier visage détecté
        face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
        
        # Chargement des encodages existants
        known_face_encodings = np.load(ENCODINGS_FILE, allow_pickle=True).tolist()
        known_face_names = np.load(NAMES_FILE, allow_pickle=True).tolist()
        
        # Vérification si l'encodage existe déjà
        if known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                os.remove(filepath)
                return jsonify({
                    'success': False,
                    'message': 'Ce visage est déjà enregistré dans le système'
                }), 400
        
        # Sauvegarde des nouveaux encodages
        known_face_encodings.append(face_encoding)
        np.save(ENCODINGS_FILE, np.array(known_face_encodings, dtype=object))
        
        return jsonify({
            'success': True,
            'message': 'Image uploadée et visage encodé avec succès',
            'filename': filename,
            'path': f'/uploads/images/{filename}'
        })
        
    except Exception as e:
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False, 'message': str(e)}), 500
    
# Route pour servir les images
@app.route('/uploads/images/<filename>')
def serve_image(filename):
    """Servir les images uploadées"""
    try:
        return send_file(os.path.join(UPLOAD_FOLDER, filename))
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'Image non trouvée'}), 404
    
@app.route('/api/absent', methods=['GET'])
def get_absents():
    """Récupérer la liste des personnes absentes"""
    try:
        absents = load_json_file(ABSENT_FILE)
        return jsonify({
            'success': True,
            'data': absents,
            'total': len(absents)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/api/absent/<person_id>', methods=['PUT'])
def update_absent_reason(person_id):
    """Mettre à jour la raison d'absence d'une personne"""
    try:
        data = request.get_json()
        absents = load_json_file(ABSENT_FILE)
        
        # Trouver l'absence à mettre à jour
        for absent in absents:
            if absent['id'] == person_id:
                absent['raison'] = data.get('raison', '')
                break
        
        if save_json_file(ABSENT_FILE, absents):
            return jsonify({
                'success': True, 
                'message': 'Raison mise à jour avec succès'
            })
        else:
            return jsonify({
                'success': False, 
                'message': 'Erreur lors de la sauvegarde'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)