from flask import Flask, request, jsonify
import os
import sys
from flask_cors import CORS
from clustering import detect_clusters
import uuid
from werkzeug.utils import secure_filename
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from models import db, Case, User, NGO, NGONotification, Pet, AdoptionRequest

# Let Flask find files inside model/ - must come BEFORE importing from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from cnn_model import load_model, predict_image, load_general_model, is_likely_dog
from models import db, Case, User
import os


app = Flask(__name__)
CORS(app, origins=[
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://pawcare-frontend-azure.vercel.app"
])
# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cases.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'dev-only-fallback-key')
jwt = JWTManager(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained CNN model ONCE when the server starts
model = load_model('model/pawcare_model.onnx')
# general_model = load_general_model()  # Temporarily disabled - too memory-intensive for free tier hosting
CONFIDENCE_THRESHOLD = 0.60


with app.app_context():
    db.create_all()


@app.route('/')
def home():
    return "PawCare AI backend is running!"


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    original_filename = secure_filename(file.filename)
    extension = os.path.splitext(original_filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{extension}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)

   # Gate temporarily disabled - too memory-intensive for free tier hosting
    # if not is_likely_dog(general_model, filepath):
    #     return jsonify({
    #         "error": "no_dog_detected",
    #         "message": "This doesn't appear to be a photo of a dog. Please upload a clear photo of the affected area."
    #     }), 422

    location = request.form.get('location')
    result = predict_image(model, filepath, use_tta=False)
    is_uncertain = result["confidence"] < CONFIDENCE_THRESHOLD

    new_case = Case(
        filename=unique_filename,
        prediction=result["prediction"],
        confidence=result["confidence"],
        is_uncertain=is_uncertain,
        location=location
    )
    db.session.add(new_case)
    db.session.commit()

    return jsonify({
        "case_id": new_case.id,
        "prediction": result["prediction"],
        "confidence": round(result["confidence"], 3),
        "is_uncertain": is_uncertain,
        "is_ambiguous": result["is_ambiguous"],
        "second_prediction": result["second_prediction"],
        "second_confidence": round(result["second_confidence"], 3) if result["second_confidence"] else None,
        "message": (
            "Low confidence — recommend in-person veterinary examination."
            if is_uncertain else
            "AI analysis complete."
        )
    })


@app.route('/cases', methods=['GET'])
def get_cases():
    cases = Case.query.order_by(Case.created_at.desc()).all()
    return jsonify([case.to_dict() for case in cases])


@app.route('/cases/<int:case_id>', methods=['GET'])
def get_case(case_id):
    case = Case.query.get(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    return jsonify(case.to_dict())


@app.route('/clusters', methods=['GET'])
def get_clusters():
    all_cases = Case.query.all()
    cases_as_dicts = [c.to_dict() for c in all_cases]
    clusters = detect_clusters(cases_as_dicts)
    return jsonify(clusters)


@app.route('/auth/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')

    if not username or not email or not password:
        return jsonify({"error": "username, email, and password are required"}), 400

    if role not in ['user', 'vet']:
        return jsonify({"error": "Invalid role"}), 400

    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({"error": "Username or email already taken"}), 409

    new_user = User(username=username, email=email, role=role)
    new_user.set_password(password)

    if role == 'vet':
        new_user.license_number = data.get('license_number')
        new_user.clinic_name = data.get('clinic_name')
        new_user.clinic_address = data.get('clinic_address')
        new_user.is_verified = False  # vets need admin approval before being trusted

    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "message": "Registered successfully" if role == 'user' else "Registered — awaiting admin verification before you can review cases",
        "user": new_user.to_dict()
    }), 201


@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401

    access_token = create_access_token(identity=str(user.id))
    return jsonify({"access_token": access_token, "user": user.to_dict()})


@app.route('/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user.to_dict())


@app.route('/cases/<int:case_id>/status', methods=['PATCH'])
@jwt_required()
def update_case_status(case_id):
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))

    case = Case.query.get(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    data = request.json
    new_status = data.get('status')
    vet_label = data.get('vet_confirmed_label')

    # Only verified vets can confirm/correct a diagnosis
    if vet_label and (user.role != 'vet' or not user.is_verified):
        return jsonify({"error": "Only verified veterinarians can confirm diagnoses"}), 403

    if new_status not in ['pending', 'vet_confirmed', 'resolved']:
        return jsonify({"error": "Invalid status"}), 400

    VALID_DISEASES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']
    if vet_label and vet_label not in VALID_DISEASES:
        return jsonify({"error": "Invalid disease label"}), 400

    case.status = new_status
    if vet_label:
        case.vet_confirmed_label = vet_label
        case.reviewed_by_id = user.id  # track which vet reviewed this

    db.session.commit()
    return jsonify(case.to_dict())
def require_admin():
    """Helper: returns the current user if they're an admin, otherwise None."""
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user or user.role != 'admin':
        return None
    return user


@app.route('/admin/pending-vets', methods=['GET'])
@jwt_required()
def get_pending_vets():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    pending_vets = User.query.filter_by(role='vet', is_verified=False).all()
    return jsonify([v.to_dict() for v in pending_vets])


@app.route('/admin/vets/<int:vet_id>/approve', methods=['POST'])
@jwt_required()
def approve_vet(vet_id):
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    vet = User.query.get(vet_id)
    if not vet or vet.role != 'vet':
        return jsonify({"error": "Vet not found"}), 404

    vet.is_verified = True
    db.session.commit()
    return jsonify({"message": f"{vet.username} approved", "user": vet.to_dict()})


@app.route('/admin/vets/<int:vet_id>/reject', methods=['POST'])
@jwt_required()
def reject_vet(vet_id):
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    vet = User.query.get(vet_id)
    if not vet or vet.role != 'vet':
        return jsonify({"error": "Vet not found"}), 404

    db.session.delete(vet)
    db.session.commit()
    return jsonify({"message": f"{vet.username}'s application was rejected and removed"})
from models import db, Case, User, NGO, NGONotification, Pet, AdoptionRequest

@app.route('/ngos', methods=['GET'])
def get_ngos():
    ngos = NGO.query.all()
    return jsonify([n.to_dict() for n in ngos])


@app.route('/ngos', methods=['POST'])
@jwt_required()
def create_ngo():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.json
    new_ngo = NGO(
        name=data.get('name'), phone=data.get('phone'),
        address=data.get('address'), lat=data.get('lat'), lng=data.get('lng')
    )
    db.session.add(new_ngo)
    db.session.commit()
    return jsonify(new_ngo.to_dict()), 201


@app.route('/ngos/<int:ngo_id>/notify', methods=['POST'])
@jwt_required()
def notify_ngo(ngo_id):
    user_id = get_jwt_identity()
    ngo = NGO.query.get(ngo_id)
    if not ngo:
        return jsonify({"error": "NGO not found"}), 404

    data = request.json or {}
    notification = NGONotification(ngo_id=ngo_id, user_id=int(user_id), message=data.get('message'))
    db.session.add(notification)
    db.session.commit()
    return jsonify({"message": f"{ngo.name} has been notified"}), 201


@app.route('/pets', methods=['GET'])
def get_pets():
    pets = Pet.query.filter_by(status='available').all()
    return jsonify([p.to_dict() for p in pets])


@app.route('/pets', methods=['POST'])
@jwt_required()
def create_pet():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.json
    new_pet = Pet(
        name=data.get('name'), breed=data.get('breed'), age=data.get('age'),
        description=data.get('description'), is_vaccinated=data.get('is_vaccinated', False)
    )
    db.session.add(new_pet)
    db.session.commit()
    return jsonify(new_pet.to_dict()), 201


@app.route('/pets/<int:pet_id>/adopt', methods=['POST'])
@jwt_required()
def request_adoption(pet_id):
    user_id = get_jwt_identity()
    pet = Pet.query.get(pet_id)
    if not pet:
        return jsonify({"error": "Pet not found"}), 404

    existing = AdoptionRequest.query.filter_by(pet_id=pet_id, user_id=int(user_id)).first()
    if existing:
        return jsonify({"error": "You've already requested to adopt this pet"}), 409

    new_request = AdoptionRequest(pet_id=pet_id, user_id=int(user_id))
    db.session.add(new_request)
    db.session.commit()
    return jsonify({"message": f"Adoption request for {pet.name} submitted"}), 201


if __name__ == '__main__':
    app.run(debug=True)