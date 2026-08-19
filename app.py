from flask import Flask, request, jsonify, send_from_directory, redirect
import os
import sys
import uuid
import shutil
from flask_cors import CORS
from clustering import detect_clusters
from werkzeug.utils import secure_filename
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies, get_csrf_token, get_jwt,
)
from models import db, Case, User, NGO, NGONotification, Pet, AdoptionRequest
from datetime import datetime, timedelta
from PIL import Image

# Dynamic absolute directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
sys.path.append(MODEL_DIR)

from cnn_model import load_model, predict_image, load_general_model, is_likely_dog
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

app = Flask(__name__)

# CORS configuration
CORS(app, origins=[
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "https://pawcare-frontend-azure.vercel.app"
], supports_credentials=True)

# Database configuration
database_url = os.environ.get('DATABASE_URL', 'sqlite:///cases.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# JWT Secret configuration with safe production fallback
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    JWT_SECRET_KEY = 'pawcare-production-default-secret-key-32-chars-min'
    print("[startup warning] JWT_SECRET_KEY not set in environment; using fallback key.")
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY

# JWT delivered via httpOnly cookie
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_COOKIE_CSRF_PROTECT'] = True
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_COOKIE_SECURE'] = os.environ.get('JWT_COOKIE_SECURE', 'true').lower() != 'false'
app.config['JWT_COOKIE_SAMESITE'] = 'None' if app.config['JWT_COOKIE_SECURE'] else 'Lax'

jwt = JWTManager(app)

# JWT Error Handlers returning structured JSON instead of 500 crashes
@jwt.unauthorized_loader
def custom_unauthorized_response(err_str):
    return jsonify({"error": "unauthorized", "message": err_str}), 401

@jwt.invalid_token_loader
def custom_invalid_token_response(err_str):
    return jsonify({"error": "invalid_token", "message": err_str}), 401

@jwt.expired_token_loader
def custom_expired_token_response(jwt_header, jwt_payload):
    return jsonify({"error": "token_expired", "message": "Session expired. Please log in again."}), 401

@app.errorhandler(500)
def handle_500_error(error):
    return jsonify({"error": "internal_server_error", "message": "An internal server error occurred."}), 500

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Upload Safety Limits
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_IMAGE_MIME_TYPES = {'image/png', 'image/jpeg', 'image/webp'}
MIN_IMAGE_DIMENSION = 64      # px
MAX_IMAGE_DIMENSION = 6000    # px


def allowed_extension(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def validate_image_file(file_storage):
    if not file_storage or not file_storage.filename:
        return False, "No file provided"

    if not allowed_extension(file_storage.filename):
        return False, "Unsupported file type. Allowed: PNG, JPG, JPEG, WEBP"

    if file_storage.mimetype and file_storage.mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        return False, "Unsupported file type"

    try:
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        img.verify()

        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        width, height = img.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return False, f"Image too small (minimum {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}px)"
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Image too large (maximum {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}px)"

        file_storage.stream.seek(0)
    except Exception:
        return False, "File is not a valid image"

    return True, None


# Load ONNX models with absolute path resolution
MODEL_PATH = os.path.join(MODEL_DIR, 'pawcare_model.onnx')
GENERAL_MODEL_PATH = os.path.join(MODEL_DIR, 'general_imagenet_model.onnx')

model = load_model(MODEL_PATH)
general_model = load_general_model(GENERAL_MODEL_PATH)
CONFIDENCE_THRESHOLD = float(os.environ.get('AI_CONFIDENCE_THRESHOLD', '0.65'))


with app.app_context():
    db.create_all()

    # Auto-migration: ensure newly added columns exist in case table across SQLite and Postgres
    try:
        with db.engine.connect() as conn:
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN reported_by_id INTEGER REFERENCES "user"(id)'))
                conn.commit()
                print("[startup] Added missing reported_by_id column to case table.")
            except Exception:
                pass
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN reviewed_by_id INTEGER REFERENCES "user"(id)'))
                conn.commit()
            except Exception:
                pass
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN vet_confirmed_label VARCHAR(100)'))
                conn.commit()
            except Exception:
                pass
    except Exception as mig_err:
        print(f"[startup] Migration notice: {mig_err}")

    # Admin Auto-Seed
    FIXED_ADMIN_USERNAME = os.environ.get('FIXED_ADMIN_USERNAME')
    FIXED_ADMIN_PASSWORD = os.environ.get('FIXED_ADMIN_PASSWORD')
    FIXED_ADMIN_EMAIL = os.environ.get('FIXED_ADMIN_EMAIL', 'admin@pawcare.local')

    if FIXED_ADMIN_USERNAME and FIXED_ADMIN_PASSWORD:
        existing_admin = User.query.filter_by(username=FIXED_ADMIN_USERNAME).first()
        if not existing_admin:
            new_admin = User(
                username=FIXED_ADMIN_USERNAME,
                email=FIXED_ADMIN_EMAIL,
                role='admin',
                is_verified=True,
            )
            new_admin.set_password(FIXED_ADMIN_PASSWORD)
            db.session.add(new_admin)
            db.session.commit()
            print(f"[startup] Created permanent admin account: {FIXED_ADMIN_USERNAME}")
        elif existing_admin.role != 'admin':
            existing_admin.role = 'admin'
            existing_admin.is_verified = True
            db.session.commit()
            print(f"[startup] Promoted existing account to admin: {FIXED_ADMIN_USERNAME}")
        else:
            print(f"[startup] Admin account already exists: {FIXED_ADMIN_USERNAME}")
    else:
        print("[startup] FIXED_ADMIN_USERNAME / FIXED_ADMIN_PASSWORD not set — skipping admin auto-seed.")


def get_current_user_obj():
    """Helper: load the User row for the current JWT identity safely, or None."""
    try:
        from flask_jwt_extended import get_jwt_identity
        user_id = get_jwt_identity()
        if not user_id:
            return None
        return db.session.get(User, int(user_id))
    except Exception:
        return None


def require_admin():
    """Helper: returns the current user if they're an admin, otherwise None."""
    user = get_current_user_obj()
    if not user or user.role != 'admin':
        return None
    return user


def require_verified_vet_or_admin():
    """Helper: returns the current user if they're an admin or a verified vet."""
    user = get_current_user_obj()
    if not user:
        return None
    if user.role == 'admin':
        return user
    if user.role == 'vet' and user.is_verified:
        return user
    return None


@app.route('/')
def home():
    return "PawCare AI backend is running!"


@app.route('/upload', methods=['POST'])
@app.route('/api/upload', methods=['POST'])
def upload():
    # Safely check if the request contains valid user credentials without failing guest uploads
    user_id = None
    try:
        from flask_jwt_extended import verify_jwt_in_request
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
    except Exception:
        user_id = None


    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    is_valid, error_message = validate_image_file(file)
    if not is_valid:
        return jsonify({"error": "invalid_image", "message": error_message}), 400

    # Save temporarily to disk for model inference
    original_filename = secure_filename(file.filename or 'upload.jpg')
    extension = os.path.splitext(original_filename)[1] or '.jpg'
    temp_filename = f"{uuid.uuid4().hex}{extension}"
    temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.stream.seek(0)
    file.save(temp_filepath)

    try:
        if not is_likely_dog(general_model, temp_filepath):
            return jsonify({
                "error": "no_dog_detected",
                "message": "This doesn't appear to be a photo of a dog. Please upload a clear photo of the affected area."
            }), 422

        # Run prediction on the local temp file
        location = request.form.get('location')
        result = predict_image(model, temp_filepath, use_tta=False)
        is_uncertain = result["confidence"] < CONFIDENCE_THRESHOLD

        case_id = None
        # Save case and upload to Cloudinary (with local storage fallback) if user is authenticated
        if user_id:
            image_url = None
            if os.environ.get('CLOUDINARY_API_KEY') and os.environ.get('CLOUDINARY_CLOUD_NAME'):
                try:
                    with open(temp_filepath, 'rb') as f:
                        upload_result = cloudinary.uploader.upload(
                            f,
                            folder='pawcare/cases',
                            resource_type='auto'
                        )
                    image_url = upload_result.get('secure_url')
                except Exception as e:
                    print(f"[upload] Cloudinary upload exception: {e}")

            if not image_url:
                perm_filename = f"case_{temp_filename}"
                perm_filepath = os.path.join(UPLOAD_FOLDER, perm_filename)
                try:
                    shutil.copy2(temp_filepath, perm_filepath)
                    image_url = perm_filename
                except Exception:
                    image_url = temp_filename

            try:
                valid_user = db.session.get(User, int(user_id)) if user_id else None
                reported_id = valid_user.id if valid_user else None

                new_case = Case(
                    filename=image_url,
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    is_uncertain=is_uncertain,
                    location=location,
                    reported_by_id=reported_id,
                )
                db.session.add(new_case)
                db.session.commit()
                case_id = new_case.id
            except Exception as db_err:
                db.session.rollback()
                print(f"[upload] Database save failed: {db_err}")

        return jsonify({
            "case_id": case_id,
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
    except Exception as inference_err:
        print(f"[upload] Error during inference/processing: {inference_err}")
        return jsonify({"error": "processing_failed", "message": f"Failed to analyze image: {str(inference_err)}"}), 500
    finally:
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except Exception:
                pass


@app.route('/uploads/<filename>', methods=['GET'])
@app.route('/api/uploads/<filename>', methods=['GET'])
def serve_upload(filename):
    """Backwards compatibility upload route."""
    if filename.startswith('http'):
        return redirect(filename)
    safe_filename = secure_filename(filename)
    return send_from_directory(UPLOAD_FOLDER, safe_filename)


@app.route('/cases', methods=['GET'])
@app.route('/api/cases', methods=['GET'])
@jwt_required()
def get_cases():
    user = get_current_user_obj()
    if not user:
        return jsonify({"error": "User not found"}), 404

    if user.role in ('vet', 'admin'):
        cases = Case.query.order_by(Case.created_at.desc()).all()
    else:
        cases = Case.query.filter_by(reported_by_id=user.id).order_by(Case.created_at.desc()).all()

    return jsonify([case.to_dict() for case in cases])


@app.route('/cases/<int:case_id>', methods=['GET'])
@app.route('/api/cases/<int:case_id>', methods=['GET'])
@jwt_required()
def get_case(case_id):
    user = get_current_user_obj()
    if not user:
        return jsonify({"error": "User not found"}), 404

    case = db.session.get(Case, case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    if user.role not in ('vet', 'admin') and case.reported_by_id != user.id:
        return jsonify({"error": "You don't have access to this case"}), 403

    return jsonify(case.to_dict())


@app.route('/clusters', methods=['GET'])
@app.route('/api/clusters', methods=['GET'])
@jwt_required()
def get_clusters():
    all_cases = Case.query.all()
    cases_as_dicts = [c.to_dict() for c in all_cases]
    clusters = detect_clusters(cases_as_dicts)
    return jsonify(clusters)


@app.route('/auth/register', methods=['POST'])
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json or {}
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
        new_user.is_verified = False

    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "message": "Registered successfully" if role == 'user' else "Registered — awaiting admin verification before you can review cases",
        "user": new_user.to_dict()
    }), 201


@app.route('/auth/login', methods=['POST'])
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401

    access_token = create_access_token(identity=str(user.id))
    csrf_token = get_csrf_token(access_token)
    resp = jsonify({"user": user.to_dict(), "csrf_token": csrf_token})
    set_access_cookies(resp, access_token)
    return resp


@app.route('/auth/logout', methods=['POST'])
@app.route('/api/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    resp = jsonify({"message": "Logged out"})
    unset_jwt_cookies(resp)
    return resp


@app.route('/auth/me', methods=['GET'])
@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        user_id = get_jwt_identity()
        if not user_id:
            return jsonify({"error": "Unauthorized"}), 401
        user = db.session.get(User, int(user_id))
        if not user:
            return jsonify({"error": "User not found"}), 404
        raw_jwt = get_jwt() or {}
        csrf_token = raw_jwt.get("csrf")
        return jsonify({**user.to_dict(), "csrf_token": csrf_token})
    except Exception as e:
        return jsonify({"error": "session_lookup_failed", "message": str(e)}), 500


@app.route('/cases/<int:case_id>/status', methods=['PATCH'])
@app.route('/api/cases/<int:case_id>/status', methods=['PATCH'])
@jwt_required()
def update_case_status(case_id):
    user = require_verified_vet_or_admin()
    if not user:
        return jsonify({"error": "Only verified veterinarians or admins can update case status"}), 403

    case = db.session.get(Case, case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    data = request.json or {}
    new_status = data.get('status')
    vet_label = data.get('vet_confirmed_label')

    if new_status not in ['pending', 'vet_confirmed', 'resolved']:
        return jsonify({"error": "Invalid status"}), 400

    VALID_DISEASES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']
    if vet_label and vet_label not in VALID_DISEASES:
        return jsonify({"error": "Invalid disease label"}), 400

    case.status = new_status
    if vet_label:
        case.vet_confirmed_label = vet_label
        case.reviewed_by_id = user.id

    db.session.commit()
    return jsonify(case.to_dict())


@app.route('/admin/pending-vets', methods=['GET'])
@app.route('/api/admin/pending-vets', methods=['GET'])
@jwt_required()
def get_pending_vets():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    pending_vets = User.query.filter_by(role='vet', is_verified=False).all()
    return jsonify([v.to_dict() for v in pending_vets])


@app.route('/admin/vets/<int:vet_id>/approve', methods=['POST'])
@app.route('/api/admin/vets/<int:vet_id>/approve', methods=['POST'])
@jwt_required()
def approve_vet(vet_id):
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    vet = db.session.get(User, vet_id)
    if not vet or vet.role != 'vet':
        return jsonify({"error": "Vet not found"}), 404

    vet.is_verified = True
    db.session.commit()
    return jsonify({"message": f"{vet.username} approved", "user": vet.to_dict()})


@app.route('/admin/vets/<int:vet_id>/reject', methods=['POST'])
@app.route('/api/admin/vets/<int:vet_id>/reject', methods=['POST'])
@jwt_required()
def reject_vet(vet_id):
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    vet = db.session.get(User, vet_id)
    if not vet or vet.role != 'vet':
        return jsonify({"error": "Vet not found"}), 404

    vet.role = 'user'
    vet.is_verified = False
    db.session.commit()
    return jsonify({"message": f"{vet.username}'s vet application was rejected. Their account remains active as a regular user."})


@app.route('/admin/export-corrections', methods=['GET'])
@app.route('/api/admin/export-corrections', methods=['GET'])
@jwt_required()
def export_corrections():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    corrected_cases = Case.query.filter(Case.vet_confirmed_label.isnot(None)).all()

    export_data = []
    for case in corrected_cases:
        export_data.append({
            "case_id": case.id,
            "image_url": case.filename if case.filename.startswith('http')
                         else f"/uploads/{case.filename}",
            "ai_prediction": case.prediction,
            "ai_confidence": case.confidence,
            "vet_confirmed_label": case.vet_confirmed_label,
            "ai_was_correct": case.prediction == case.vet_confirmed_label,
            "reviewed_by_id": case.reviewed_by_id,
        })

    agreement_count = sum(1 for c in export_data if c["ai_was_correct"])
    total = len(export_data)

    return jsonify({
        "total_vet_reviewed_cases": total,
        "ai_agreement_rate": round(agreement_count / total, 3) if total > 0 else None,
        "cases": export_data
    })


@app.route('/ngos', methods=['GET'])
@app.route('/api/ngos', methods=['GET'])
def get_ngos():
    ngos = NGO.query.all()
    return jsonify([n.to_dict() for n in ngos])


@app.route('/ngos', methods=['POST'])
@app.route('/api/ngos', methods=['POST'])
@jwt_required()
def create_ngo():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.json or {}
    new_ngo = NGO(
        name=data.get('name'), phone=data.get('phone'),
        address=data.get('address'), lat=data.get('lat'), lng=data.get('lng')
    )
    db.session.add(new_ngo)
    db.session.commit()
    return jsonify(new_ngo.to_dict()), 201


NGO_NOTIFY_COOLDOWN = timedelta(minutes=5)
MAX_NOTIFY_MESSAGE_LENGTH = 500


@app.route('/ngos/<int:ngo_id>/notify', methods=['POST'])
@app.route('/api/ngos/<int:ngo_id>/notify', methods=['POST'])
@jwt_required()
def notify_ngo(ngo_id):
    user_id = get_jwt_identity()
    ngo = db.session.get(NGO, ngo_id)
    if not ngo:
        return jsonify({"error": "NGO not found"}), 404

    data = request.json or {}
    message = (data.get('message') or '').strip()
    if not message:
        return jsonify({"error": "message is required"}), 400
    if len(message) > MAX_NOTIFY_MESSAGE_LENGTH:
        return jsonify({"error": f"message must be {MAX_NOTIFY_MESSAGE_LENGTH} characters or fewer"}), 400

    recent_cutoff = datetime.utcnow() - NGO_NOTIFY_COOLDOWN
    recent_notification = NGONotification.query.filter(
        NGONotification.ngo_id == ngo_id,
        NGONotification.user_id == int(user_id),
        NGONotification.created_at >= recent_cutoff,
    ).first()
    if recent_notification:
        return jsonify({"error": "You've already notified this NGO recently. Please wait a few minutes before trying again."}), 429

    notification = NGONotification(ngo_id=ngo_id, user_id=int(user_id), message=message)
    db.session.add(notification)
    db.session.commit()
    return jsonify({"message": f"{ngo.name} has been notified"}), 201


@app.route('/pets', methods=['GET'])
@app.route('/api/pets', methods=['GET'])
def get_pets():
    pets = Pet.query.filter_by(status='available').all()
    return jsonify([p.to_dict() for p in pets])


@app.route('/pets/<int:pet_id>/adopt', methods=['POST'])
@app.route('/api/pets/<int:pet_id>/adopt', methods=['POST'])
@jwt_required()
def request_adoption(pet_id):
    user_id = get_jwt_identity()
    pet = db.session.get(Pet, pet_id)
    if not pet:
        return jsonify({"error": "Pet not found"}), 404

    existing = AdoptionRequest.query.filter_by(pet_id=pet_id, user_id=int(user_id)).first()
    if existing:
        return jsonify({"error": "You've already requested to adopt this pet"}), 409

    new_request = AdoptionRequest(pet_id=pet_id, user_id=int(user_id))
    db.session.add(new_request)
    db.session.commit()
    return jsonify({"message": f"Adoption request for {pet.name} submitted"}), 201


@app.route('/pets', methods=['POST'])
@app.route('/api/pets', methods=['POST'])
@jwt_required()
def create_pet():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.form if request.form else (request.json or {})

    image_url = None
    if 'image' in request.files and request.files['image'].filename:
        file = request.files['image']

        is_valid, error_message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": "invalid_image", "message": error_message}), 400

        try:
            result = cloudinary.uploader.upload(
                file,
                folder='pawcare/pets',
                resource_type='auto'
            )
            image_url = result.get('secure_url')
        except Exception as e:
            print(f"[create_pet] Cloudinary upload error: {e}")

    is_vaccinated_raw = data.get('is_vaccinated', False)
    is_vaccinated = is_vaccinated_raw in (True, 'true', 'True', '1', 1)

    new_pet = Pet(
        name=data.get('name'),
        breed=data.get('breed'),
        age=data.get('age'),
        description=data.get('description'),
        is_vaccinated=is_vaccinated,
        image_filename=image_url,
        status=data.get('status', 'available'),
        created_at=datetime.utcnow(),
    )
    db.session.add(new_pet)
    db.session.commit()
    return jsonify(new_pet.to_dict()), 201


if __name__ == '__main__':
    app.run(debug=True)