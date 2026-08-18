from flask import Flask, request, jsonify, send_from_directory, redirect
import os
import sys
from flask_cors import CORS
from clustering import detect_clusters
import uuid
from werkzeug.utils import secure_filename
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies, get_csrf_token, get_jwt,
)
from models import db, Case, User, NGO, NGONotification, Pet, AdoptionRequest
from datetime import datetime, timedelta
from PIL import Image
# Let Flask find files inside model/ - must come BEFORE importing from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))
from cnn_model import load_model, predict_image, load_general_model, is_likely_dog
import cloudinary
import cloudinary.uploader

cloudinary.config(
    cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
    api_key=os.environ.get('CLOUDINARY_API_KEY'),
    api_secret=os.environ.get('CLOUDINARY_API_SECRET')
)

app = Flask(__name__)
# supports_credentials=True is required for cookie-based JWTs — without it
# the browser will not send/accept the auth cookie on cross-origin requests
# (frontend on vercel.app, backend on onrender.com are different origins).
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

JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
if not JWT_SECRET_KEY:
    if os.environ.get('FLASK_ENV') == 'development' or app.debug:
        JWT_SECRET_KEY = 'dev-only-fallback-key'
    else:
        raise RuntimeError("JWT_SECRET_KEY environment variable is required in production")
app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY

# ── JWT delivered via httpOnly cookie instead of the response body ─────────
# Previously the token was returned as JSON and the frontend stored it in
# localStorage, which any injected/XSS script on the page could read
# directly. It's now set as an httpOnly cookie — JavaScript can't read it at
# all — with a separate, non-httpOnly CSRF cookie the frontend must echo
# back as a header on state-changing requests (the standard "double submit"
# pattern flask-jwt-extended implements for you).
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_COOKIE_CSRF_PROTECT'] = True
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
# Frontend and backend are on different domains in production, so the
# cookie must be SameSite=None + Secure to be sent cross-site at all — that
# combination only works over HTTPS. Set JWT_COOKIE_SECURE=false in your
# LOCAL dev .env only if you're testing over plain http://localhost.
app.config['JWT_COOKIE_SECURE'] = os.environ.get('JWT_COOKIE_SECURE', 'true').lower() != 'false'
app.config['JWT_COOKIE_SAMESITE'] = 'None' if app.config['JWT_COOKIE_SECURE'] else 'Lax'

jwt = JWTManager(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── UPLOAD SAFETY LIMITS ────────────────────────────────────────────────
# Reject anything over 10MB at the Flask level before it's even fully
# read into memory (Flask returns 413 automatically once this is set).
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_IMAGE_MIME_TYPES = {'image/png', 'image/jpeg', 'image/webp'}
MIN_IMAGE_DIMENSION = 64      # px, reject tiny/garbage images
MAX_IMAGE_DIMENSION = 6000    # px, reject decompression-bomb-style images


def allowed_extension(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def validate_image_file(file_storage):
    """
    Validates an uploaded file is actually a real, reasonably-sized image
    before we do anything else with it (save to disk, run inference,
    upload to Cloudinary). Returns (is_valid, error_message).

    Checks, in order:
      1. A filename was actually provided
      2. Extension is on the allowlist
      3. Declared MIME type is on the allowlist (defense in depth — this
         can be spoofed by the client, so it's not relied on alone)
      4. Pillow can actually decode it as an image (catches renamed
         non-image files, corrupted files, zip bombs disguised as images)
      5. Image dimensions are within sane bounds
    """
    if not file_storage or not file_storage.filename:
        return False, "No file provided"

    if not allowed_extension(file_storage.filename):
        return False, "Unsupported file type. Allowed: PNG, JPG, JPEG, WEBP"

    if file_storage.mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        return False, "Unsupported file type"

    try:
        # verify() reads the file to check it's a valid, non-corrupt image
        # without decoding full pixel data — but it consumes the stream,
        # so we need to seek back to 0 afterwards to actually use the file.
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        img.verify()

        # verify() invalidates the Image object, so re-open to check dimensions
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


# Load the trained CNN model ONCE when the server starts
model = load_model('model/pawcare_model.onnx')
general_model = load_general_model('model/general_imagenet_model.onnx')  # Re-enabled: ONNX version is lightweight
CONFIDENCE_THRESHOLD = 0.60


with app.app_context():
    db.create_all()

    # ── PERMANENT ADMIN AUTO-SEED ──────────────────────────────────────
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
    # ─────────────────────────────────────────────────────────────────────


def get_current_user_obj():
    """Helper: load the User row for the current JWT identity, or None."""
    user_id = get_jwt_identity()
    if not user_id:
        return None
    return User.query.get(int(user_id))


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
@jwt_required(optional=True)
def upload():
    user_id = get_jwt_identity()

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    is_valid, error_message = validate_image_file(file)
    if not is_valid:
        return jsonify({"error": "invalid_image", "message": error_message}), 400

    # Save temporarily to disk for the dog detection check
    original_filename = secure_filename(file.filename)
    extension = os.path.splitext(original_filename)[1]
    temp_filename = f"{uuid.uuid4().hex}{extension}"
    temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
    file.save(temp_filepath)

    if not is_likely_dog(general_model, temp_filepath):
        os.remove(temp_filepath)
        return jsonify({
            "error": "no_dog_detected",
            "message": "This doesn't appear to be a photo of a dog. Please upload a clear photo of the affected area."
        }), 422

    # Dog check passed, now upload to Cloudinary
    try:
        with open(temp_filepath, 'rb') as f:
            result = cloudinary.uploader.upload(
                f,
                folder='pawcare/cases',
                resource_type='auto'
            )
        image_url = result['secure_url']
    except Exception as e:
        os.remove(temp_filepath)
        return jsonify({"error": f"Failed to upload image: {str(e)}"}), 400

    # Run prediction on the local temp file (not the URL)
    location = request.form.get('location')
    result = predict_image(model, temp_filepath, use_tta=False)  # Pass local temp file, not URL
    is_uncertain = result["confidence"] < CONFIDENCE_THRESHOLD

    # Now that prediction is done, clean up the temp file
    os.remove(temp_filepath)

    new_case = Case(
        filename=image_url,  # Store Cloudinary URL in database
        prediction=result["prediction"],
        confidence=result["confidence"],
        is_uncertain=is_uncertain,
        location=location,
        reported_by_id=int(user_id) if user_id else None,
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


@app.route('/uploads/<filename>', methods=['GET'])
def serve_upload(filename):
    """
    Deprecated: photos now stored on Cloudinary.
    This route kept for backwards compatibility only.
    """
    if filename.startswith('http'):
        return redirect(filename)

    safe_filename = secure_filename(filename)
    return send_from_directory(UPLOAD_FOLDER, safe_filename)


@app.route('/cases', methods=['GET'])
@jwt_required()
def get_cases():
    user = get_current_user_obj()
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Vets and admins need to see every case to do their jobs (review /
    # verify diagnoses, manage outbreaks). Regular users only ever see
    # the cases they personally reported.
    if user.role in ('vet', 'admin'):
        cases = Case.query.order_by(Case.created_at.desc()).all()
    else:
        cases = Case.query.filter_by(reported_by_id=user.id).order_by(Case.created_at.desc()).all()

    return jsonify([case.to_dict() for case in cases])


@app.route('/cases/<int:case_id>', methods=['GET'])
@jwt_required()
def get_case(case_id):
    user = get_current_user_obj()
    if not user:
        return jsonify({"error": "User not found"}), 404

    case = Case.query.get(case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    if user.role not in ('vet', 'admin') and case.reported_by_id != user.id:
        return jsonify({"error": "You don't have access to this case"}), 403

    return jsonify(case.to_dict())


@app.route('/clusters', methods=['GET'])
@jwt_required()
def get_clusters():
    # Cluster data aggregates locations/diseases across ALL users' cases,
    # so this stays behind login (any logged-in role) rather than fully
    # public, but isn't restricted to vet/admin like /cases is.
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
    csrf_token = get_csrf_token(access_token)
    resp = jsonify({"user": user.to_dict(), "csrf_token": csrf_token})
    # Sets both the httpOnly JWT cookie and the readable CSRF cookie.
    # No token in the JSON body anymore — nothing for client-side JS to
    # read or accidentally leak.
    set_access_cookies(resp, access_token)
    return resp


@app.route('/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    resp = jsonify({"message": "Logged out"})
    unset_jwt_cookies(resp)
    return resp


@app.route('/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    user_id = get_jwt_identity()
    user = User.query.get(int(user_id))
    if not user:
        return jsonify({"error": "User not found"}), 404
    raw_jwt = get_jwt()
    csrf_token = raw_jwt.get("csrf")
    return jsonify({**user.to_dict(), "csrf_token": csrf_token})


@app.route('/cases/<int:case_id>/status', methods=['PATCH'])
@jwt_required()
def update_case_status(case_id):
    # Only verified vets or admins may change a case's status or diagnosis
    # at all — this used to only gate the vet_confirmed_label field, which
    # meant any logged-in regular user could still change `status` freely
    # (e.g. flip a case straight to "resolved") as long as they didn't
    # also send a vet_confirmed_label.
    user = require_verified_vet_or_admin()
    if not user:
        return jsonify({"error": "Only verified veterinarians or admins can update case status"}), 403

    case = Case.query.get(case_id)
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
        case.reviewed_by_id = user.id  # track which vet reviewed this

    db.session.commit()
    return jsonify(case.to_dict())


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

    vet.role = 'user'
    vet.is_verified = False
    db.session.commit()
    return jsonify({"message": f"{vet.username}'s vet application was rejected. Their account remains active as a regular user."})


@app.route('/admin/export-corrections', methods=['GET'])
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
            # case.filename is already a full Cloudinary URL (post-migration) —
            # previously this prepended "/uploads/" to it, producing a broken
            # path like "/uploads/https://res.cloudinary.com/...". Old rows
            # created before the Cloudinary migration may still hold a bare
            # local filename, so we only add the legacy prefix in that case.
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


NGO_NOTIFY_COOLDOWN = timedelta(minutes=5)
MAX_NOTIFY_MESSAGE_LENGTH = 500


@app.route('/ngos/<int:ngo_id>/notify', methods=['POST'])
@jwt_required()
def notify_ngo(ngo_id):
    # Still intentionally open to any authenticated user (any user should
    # be able to flag a nearby outbreak to an NGO) but now with basic
    # input validation and an anti-spam cooldown, since previously there
    # was no limit on how often or with what content someone could hit
    # this endpoint.
    user_id = get_jwt_identity()
    ngo = NGO.query.get(ngo_id)
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
def get_pets():
    pets = Pet.query.filter_by(status='available').all()
    return jsonify([p.to_dict() for p in pets])


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


@app.route('/pets', methods=['POST'])
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
            image_url = result['secure_url']
        except Exception as e:
            return jsonify({"error": f"Failed to upload image: {str(e)}"}), 400

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