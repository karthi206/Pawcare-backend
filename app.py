from flask import Flask, request, jsonify, send_from_directory, redirect
import os
import sys
import uuid
import shutil
import math
import json
import urllib.request
import urllib.parse
from flask_cors import CORS
from clustering import detect_clusters
from werkzeug.utils import secure_filename
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies, get_csrf_token, get_jwt,
)
from jwt.exceptions import (
    PyJWTError,
    InvalidTokenError,
    ExpiredSignatureError,
    DecodeError,
)
from flask_jwt_extended.exceptions import (
    JWTExtendedException,
    CSRFError,
    NoAuthorizationError,
    InvalidHeaderError,
    InvalidQueryParamError,
    JWTDecodeError,
    RevokedTokenError,
    FreshTokenRequired,
    UserLookupError,
    WrongTokenError,
)
from models import db, Case, User, NGO, NGONotification, Pet, AdoptionRequest
from datetime import datetime, timedelta
from PIL import Image

# Dynamic absolute directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
sys.path.append(MODEL_DIR)

# Auto-load .env file if present
ENV_PATH = os.path.join(BASE_DIR, '.env')
if os.path.isfile(ENV_PATH):
    try:
        with open(ENV_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
    except Exception as env_err:
        print(f"[startup] Notice reading .env: {env_err}")

from cnn_model import load_model, predict_image, load_general_model, is_likely_dog, load_ood_reference

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

# JWT delivered via httpOnly cookie (and supports Authorization header)
app.config['JWT_TOKEN_LOCATION'] = ['cookies', 'headers']
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_COOKIE_CSRF_PROTECT'] = True
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_COOKIE_SECURE'] = os.environ.get('JWT_COOKIE_SECURE', 'true').lower() != 'false'
app.config['JWT_COOKIE_SAMESITE'] = 'None' if app.config['JWT_COOKIE_SECURE'] else 'Lax'

jwt = JWTManager(app)

# Flask-JWT-Extended Loader Callbacks returning structured JSON instead of 500 crashes
@jwt.unauthorized_loader
def custom_unauthorized_response(err_str):
    return jsonify({"error": "unauthorized", "message": err_str or "Missing or invalid authorization token."}), 401

@jwt.invalid_token_loader
def custom_invalid_token_response(err_str):
    return jsonify({"error": "invalid_token", "message": err_str or "Invalid token."}), 401

@jwt.expired_token_loader
def custom_expired_token_response(jwt_header=None, jwt_payload=None):
    return jsonify({"error": "token_expired", "message": "Session expired. Please log in again."}), 401

@jwt.needs_fresh_token_loader
def custom_needs_fresh_token_response(jwt_header=None, jwt_payload=None):
    return jsonify({"error": "fresh_token_required", "message": "Fresh token required. Please log in again."}), 401

@jwt.revoked_token_loader
def custom_revoked_token_response(jwt_header=None, jwt_payload=None):
    return jsonify({"error": "token_revoked", "message": "Token has been revoked. Please log in again."}), 401

@jwt.token_verification_failed_loader
def custom_token_verification_failed_response(jwt_header=None, jwt_payload=None):
    return jsonify({"error": "token_verification_failed", "message": "User claims verification failed."}), 400

@jwt.user_lookup_error_loader
def custom_user_lookup_error_response(jwt_header=None, jwt_payload=None):
    return jsonify({"error": "user_lookup_failed", "message": "User associated with token not found."}), 401


# Global Exception Handlers for JWT, PyJWT, and CSRF Exceptions to prevent 500 crashes
@app.errorhandler(ExpiredSignatureError)
def handle_expired_signature_error(e):
    return jsonify({"error": "token_expired", "message": "Session expired. Please log in again."}), 401

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    return jsonify({"error": "csrf_error", "message": str(e) or "CSRF token validation failed."}), 401

@app.errorhandler(NoAuthorizationError)
def handle_no_authorization_error(e):
    return jsonify({"error": "unauthorized", "message": str(e) or "Authorization required."}), 401

@app.errorhandler(InvalidTokenError)
@app.errorhandler(DecodeError)
@app.errorhandler(JWTDecodeError)
@app.errorhandler(InvalidHeaderError)
@app.errorhandler(InvalidQueryParamError)
@app.errorhandler(WrongTokenError)
def handle_invalid_token_error(e):
    return jsonify({"error": "invalid_token", "message": str(e) or "Invalid token."}), 401

@app.errorhandler(RevokedTokenError)
def handle_revoked_token_error(e):
    return jsonify({"error": "token_revoked", "message": "Token has been revoked. Please log in again."}), 401

@app.errorhandler(FreshTokenRequired)
def handle_fresh_token_error(e):
    return jsonify({"error": "fresh_token_required", "message": "Fresh token required. Please log in again."}), 401

@app.errorhandler(UserLookupError)
def handle_user_lookup_error(e):
    return jsonify({"error": "user_lookup_failed", "message": str(e) or "User lookup failed."}), 401

@app.errorhandler(JWTExtendedException)
def handle_jwt_extended_exception(e):
    return jsonify({"error": "invalid_token", "message": str(e) or "Token validation failed."}), 401

@app.errorhandler(PyJWTError)
def handle_pyjwt_error(e):
    return jsonify({"error": "invalid_token", "message": str(e) or "Token decoding failed."}), 401

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
# UPDATED: points to the ML v2 dual-output model (logits + pooled features),
# needed for Mahalanobis OOD detection. Replaces the old single-output
# pawcare_model.onnx. See PawCare ML v2 roadmap Step 13/14.
MODEL_PATH = os.path.join(MODEL_DIR, 'pawcare_mobilenetv2_with_features.onnx')
GENERAL_MODEL_PATH = os.path.join(MODEL_DIR, 'general_imagenet_model.onnx')
CLASS_MEANS_PATH = os.path.join(MODEL_DIR, 'class_means.npy')
COV_INV_PATH = os.path.join(MODEL_DIR, 'cov_inv.npy')

model = load_model(MODEL_PATH)
general_model = load_general_model(GENERAL_MODEL_PATH)
class_means, cov_inv = load_ood_reference(CLASS_MEANS_PATH, COV_INV_PATH)
# CONFIDENCE_THRESHOLD is now applied inside predict_image() (defaults to
# the data-justified 0.7 from cnn_model.py); no longer computed here.


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
            # NEW: prediction/confidence must now allow NULL — predict_image()
            # returns prediction=None (and confidence=None for OOD cases) for
            # the "not_recognized" and "unable_to_classify" statuses added in
            # ML v2 Step 10. Existing tables created before this change still
            # have the old NOT NULL constraint baked in, so it must be
            # dropped explicitly; db.create_all() never alters existing
            # tables. Postgres-only syntax (matches DATABASE_URL in
            # production); harmlessly no-ops on SQLite via the except below.
            try:
                conn.execute(db.text('ALTER TABLE "case" ALTER COLUMN prediction DROP NOT NULL'))
                conn.commit()
                print("[startup] Made case.prediction nullable.")
            except Exception:
                pass
            try:
                conn.execute(db.text('ALTER TABLE "case" ALTER COLUMN confidence DROP NOT NULL'))
                conn.commit()
                print("[startup] Made case.confidence nullable.")
            except Exception:
                pass
    except Exception as mig_err:
        print(f"[startup] Migration notice: {mig_err}")

    # Admin Auto-Seed
    FIXED_ADMIN_USERNAME = os.environ.get('FIXED_ADMIN_USERNAME', 'admin')
    FIXED_ADMIN_PASSWORD = os.environ.get('FIXED_ADMIN_PASSWORD', 'admin123')
    FIXED_ADMIN_EMAIL = os.environ.get('FIXED_ADMIN_EMAIL', 'admin@pawcare.local')

    if FIXED_ADMIN_USERNAME and FIXED_ADMIN_PASSWORD:
        existing_admin = User.query.filter(
            (User.username == FIXED_ADMIN_USERNAME) | (User.email == FIXED_ADMIN_EMAIL)
        ).first()
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
        else:
            existing_admin.username = FIXED_ADMIN_USERNAME
            existing_admin.email = FIXED_ADMIN_EMAIL
            existing_admin.role = 'admin'
            existing_admin.is_verified = True
            existing_admin.set_password(FIXED_ADMIN_PASSWORD)
            db.session.commit()
            print(f"[startup] Synchronized permanent admin account: {FIXED_ADMIN_USERNAME}")
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
@jwt_required()
def upload():
    user_id = get_jwt_identity()
    if not user_id:
        return jsonify({"error": "unauthorized", "message": "Authentication required to upload images."}), 401


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
        result = predict_image(model, temp_filepath, class_means, cov_inv, use_tta=False)

        # predict_image() now returns one of three statuses:
        #   "not_recognized"     -> image isn't a dog skin photo at all (OOD)
        #   "unable_to_classify" -> low-confidence prediction, rejected
        #   "possible_condition" -> confident, calibrated prediction
        if result["status"] == "not_recognized":
            return jsonify({
                "error": "not_recognized",
                "message": result["message"],
                "ood_distance": result["ood_distance"],
            }), 422

        is_uncertain = result["status"] == "unable_to_classify"

        case_id = None
        # Save case and upload to Cloudinary (Cloudinary-only storage, no local disk fallback)
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
            else:
                print("[upload] Cloudinary credentials missing in environment")

            if not image_url:
                return jsonify({
                    "error": "image_storage_failed",
                    "message": "Image storage failed, please try again."
                }), 502

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
                    "error": "database_error",
                    "message": "Failed to record case data."
                }), 500

        return jsonify({
            "case_id": case_id,
            "prediction": result["prediction"],
            "confidence": round(result["confidence"], 3) if result["confidence"] is not None else None,
            "is_uncertain": is_uncertain,
            "is_ambiguous": result["is_ambiguous"],
            "second_prediction": result["second_prediction"],
            "second_confidence": round(result["second_confidence"], 3) if result["second_confidence"] else None,
            "ood_distance": result["ood_distance"],
            "message": result["message"],
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
@jwt_required(optional=True)
def get_cases():
    user = get_current_user_obj()
    if not user:
        cases = Case.query.order_by(Case.created_at.desc()).all()
    elif user.role in ('vet', 'admin'):
        cases = Case.query.order_by(Case.created_at.desc()).all()
    else:
        cases = Case.query.filter_by(reported_by_id=user.id).order_by(Case.created_at.desc()).all()

    return jsonify([case.to_dict() for case in cases])


@app.route('/cases/<int:case_id>', methods=['GET'])
@app.route('/api/cases/<int:case_id>', methods=['GET'])
@jwt_required(optional=True)
def get_case(case_id):
    user = get_current_user_obj()
    case = db.session.get(Case, case_id)
    if not case:
        return jsonify({"error": "Case not found"}), 404

    if user and user.role not in ('vet', 'admin') and case.reported_by_id != user.id:
        return jsonify({"error": "You don't have access to this case"}), 403

    return jsonify(case.to_dict())


@app.route('/clusters', methods=['GET'])
@app.route('/api/clusters', methods=['GET'])
@jwt_required(optional=True)
def get_clusters():
    all_cases = Case.query.all()
    cases_as_dicts = [c.to_dict() for c in all_cases]
    clusters = detect_clusters(cases_as_dicts)
    return jsonify(clusters)


@app.route('/auth/register', methods=['POST'])
@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json or {}
    username = (data.get('username') or '').strip()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')
    role = (data.get('role') or 'user').strip().lower()

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
    data = request.get_json(silent=True) or {}
    identifier = (data.get('username') or data.get('email') or '').strip()
    password = data.get('password')

    if not identifier or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = User.query.filter(
        (User.username == identifier) | (User.email == identifier.lower())
    ).first()

    if not user or not user.check_password(password):
        return jsonify({"error": "Invalid username or password"}), 401

    access_token = create_access_token(identity=str(user.id))
    try:
        csrf_token = get_csrf_token(access_token)
    except Exception:
        csrf_token = None
    resp = jsonify({"user": user.to_dict(), "csrf_token": csrf_token})
    set_access_cookies(resp, access_token)
    return resp


@app.route('/auth/logout', methods=['POST'])
@app.route('/api/auth/logout', methods=['POST'])
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
            return jsonify({"error": "unauthorized", "message": "Authentication required."}), 401
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            return jsonify({"error": "invalid_token", "message": "Invalid token identity."}), 401
        user = db.session.get(User, user_id_int)
        if not user:
            return jsonify({"error": "user_not_found", "message": "User not found."}), 404
        raw_jwt = get_jwt() or {}
        csrf_token = raw_jwt.get("csrf")
        return jsonify({**user.to_dict(), "csrf_token": csrf_token})
    except Exception as e:
        return jsonify({"error": "unauthorized", "message": str(e)}), 401


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

    VALID_STATUSES = ['pending', 'vet_confirmed', 'resolved']
    if new_status and new_status not in VALID_STATUSES:
        return jsonify({"error": "Invalid status"}), 400

    VALID_DISEASES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'demodicosis', 'ringworm']
    if vet_label and vet_label not in VALID_DISEASES:
        return jsonify({"error": "Invalid disease label"}), 400

    if new_status:
        case.status = new_status
    elif vet_label:
        case.status = 'vet_confirmed'

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


def _haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in kilometers between two GPS coordinates."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    a = min(1.0, max(0.0, a))
    return 2.0 * r * math.asin(math.sqrt(a))


@app.route('/ngos', methods=['GET'])
@app.route('/api/ngos', methods=['GET'])
def get_ngos():
    ngos = NGO.query.all()
    return jsonify([n.to_dict() for n in ngos])


@app.route('/ngos/nearby', methods=['GET'])
@app.route('/api/ngos/nearby', methods=['GET'])
def get_nearby_ngos():
    lat_raw = request.args.get('lat')
    lng_raw = request.args.get('lng')
    radius_raw = request.args.get('radius_km')

    if lat_raw is None or lng_raw is None:
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    try:
        lat = float(lat_raw)
        lng = float(lng_raw)
    except (ValueError, TypeError):
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    if math.isnan(lat) or math.isinf(lat) or not (-90.0 <= lat <= 90.0):
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    if math.isnan(lng) or math.isinf(lng) or not (-180.0 <= lng <= 180.0):
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    radius_km = 50.0
    if radius_raw is not None and str(radius_raw).strip() != '':
        try:
            radius_km = float(radius_raw)
        except (ValueError, TypeError):
            return jsonify({
                "error": "invalid_parameters",
                "message": "radius_km must be a positive number"
            }), 400

        if math.isnan(radius_km) or math.isinf(radius_km) or radius_km <= 0:
            return jsonify({
                "error": "invalid_parameters",
                "message": "radius_km must be a positive number"
            }), 400

    ngos = NGO.query.all()
    nearby_ngos = []
    for ngo in ngos:
        if ngo.lat is None or ngo.lng is None:
            continue
        dist = _haversine_distance(lat, lng, ngo.lat, ngo.lng)
        if dist <= radius_km:
            ngo_data = ngo.to_dict()
            ngo_data["distance_km"] = round(dist, 2)
            nearby_ngos.append(ngo_data)

    nearby_ngos.sort(key=lambda x: x["distance_km"])
    return jsonify(nearby_ngos)


OVERPASS_CACHE = {}
OVERPASS_CACHE_TTL = timedelta(minutes=10)
MAX_OVERPASS_CACHE_ENTRIES = 200


@app.route('/ngos/live-nearby', methods=['GET'])
@app.route('/api/ngos/live-nearby', methods=['GET'])
def get_live_nearby_ngos():
    lat_raw = request.args.get('lat')
    lng_raw = request.args.get('lng')
    radius_raw = request.args.get('radius_km')

    if lat_raw is None or lng_raw is None:
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    try:
        lat = float(lat_raw)
        lng = float(lng_raw)
    except (ValueError, TypeError):
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    if math.isnan(lat) or math.isinf(lat) or not (-90.0 <= lat <= 90.0):
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    if math.isnan(lng) or math.isinf(lng) or not (-180.0 <= lng <= 180.0):
        return jsonify({
            "error": "invalid_parameters",
            "message": "Valid lat and lng query parameters are required"
        }), 400

    radius_km = 50.0
    if radius_raw is not None and str(radius_raw).strip() != '':
        try:
            radius_km = float(radius_raw)
        except (ValueError, TypeError):
            return jsonify({
                "error": "invalid_parameters",
                "message": "radius_km must be a positive number"
            }), 400

        if math.isnan(radius_km) or math.isinf(radius_km) or radius_km <= 0:
            return jsonify({
                "error": "invalid_parameters",
                "message": "radius_km must be a positive number"
            }), 400

    # Cap radius to 100km to avoid excessively large Overpass queries
    radius_km = min(radius_km, 100.0)

    # In-memory cache key based on rounded coordinates (~1.1km resolution)
    cache_key = (round(lat, 2), round(lng, 2), round(radius_km, 1))
    now = datetime.utcnow()

    if cache_key in OVERPASS_CACHE:
        cached_time, cached_data = OVERPASS_CACHE[cache_key]
        if now - cached_time < OVERPASS_CACHE_TTL:
            return jsonify(cached_data)

    radius_meters = int(radius_km * 1000)
    overpass_query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="animal_shelter"](around:{radius_meters},{lat},{lng});
      node["amenity"="veterinary"](around:{radius_meters},{lat},{lng});
      node["office"="ngo"](around:{radius_meters},{lat},{lng});
      way["amenity"="animal_shelter"](around:{radius_meters},{lat},{lng});
      way["amenity"="veterinary"](around:{radius_meters},{lat},{lng});
      way["office"="ngo"](around:{radius_meters},{lat},{lng});
    );
    out center;
    """

    results = []
    try:
        post_data = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=post_data,
            headers={
                "User-Agent": "PawCareAI/1.0 (animal-welfare-locator)",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_body = resp.read().decode("utf-8")
            osm_data = json.loads(raw_body)
            elements = osm_data.get("elements", [])
            for el in elements:
                el_lat = el.get("lat") or el.get("center", {}).get("lat")
                el_lng = el.get("lon") or el.get("center", {}).get("lon")
                if el_lat is None or el_lng is None:
                    continue

                tags = el.get("tags", {})
                amenity = tags.get("amenity")
                office = tags.get("office")
                raw_name = (tags.get("name") or "").strip()

                # Reliable tags: keep animal_shelter and veterinary as-is
                if amenity == "animal_shelter":
                    place_type = "animal_shelter"
                    type_label = "Animal Shelter"
                    street = tags.get("addr:street")
                    name = raw_name or (f"{type_label} ({street})" if street else f"{type_label} (Unlisted Name)")
                elif amenity == "veterinary":
                    place_type = "veterinary"
                    type_label = "Veterinary Clinic"
                    street = tags.get("addr:street")
                    name = raw_name or (f"{type_label} ({street})" if street else f"{type_label} (Unlisted Name)")
                elif office == "ngo":
                    # office=ngo is too generic (driving schools, unrelated trusts, generic placeholders).
                    # Rule: Drop office=ngo with no name, and only keep if name contains an animal welfare keyword.
                    if not raw_name:
                        continue

                    name_lower = raw_name.lower()
                    animal_keywords = (
                        'animal', 'animals', 'dog', 'dogs', 'cat', 'cats', 'pet', 'pets',
                        'puppy', 'puppies', 'kitten', 'kittens', 'rescue', 'shelter',
                        'veterinary', 'vet', 'wildlife', 'stray', 'strays', 'welfare',
                        'spca', 'blue cross', 'paws', 'paw', 'canine', 'feline',
                        'sanctuary', 'fauna', 'creature', 'creatures', 'humane',
                        'gaushala', 'goshala', 'gau', 'jivdaya', 'jeevdaya', 'prani', 'ahimsa'
                    )
                    if not any(kw in name_lower for kw in animal_keywords):
                        continue

                    place_type = "ngo"
                    type_label = "NGO / Rescue"
                    name = raw_name
                else:
                    continue

                addr_parts = [
                    tags.get("addr:housenumber"),
                    tags.get("addr:street"),
                    tags.get("addr:city"),
                    tags.get("addr:postcode")
                ]
                address = ", ".join([p for p in addr_parts if p]) or tags.get("address") or "Address not listed in OSM"
                phone = tags.get("phone") or tags.get("contact:phone") or None

                dist = _haversine_distance(lat, lng, el_lat, el_lng)
                if dist <= radius_km:
                    results.append({
                        "id": f"osm_{el.get('type', 'node')}_{el.get('id')}",
                        "name": name,
                        "address": address,
                        "phone": phone,
                        "lat": el_lat,
                        "lng": el_lng,
                        "type": place_type,
                        "source": "osm",
                        "distance_km": round(dist, 2)
                    })

            results.sort(key=lambda x: x["distance_km"])
    except Exception as exc:
        print(f"[live-nearby] Overpass API request notice: {exc}")

    # Cache successful results or empty results up to cache size limit
    if len(OVERPASS_CACHE) >= MAX_OVERPASS_CACHE_ENTRIES:
        # Clear older half of entries
        keys_to_delete = list(OVERPASS_CACHE.keys())[:50]
        for k in keys_to_delete:
            OVERPASS_CACHE.pop(k, None)

    OVERPASS_CACHE[cache_key] = (now, results)
    return jsonify(results)


@app.route('/ngos', methods=['POST'])
@app.route('/api/ngos', methods=['POST'])
@jwt_required()
def create_ngo():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    phone = (data.get('phone') or '').strip()
    address = (data.get('address') or '').strip()
    lat_raw = data.get('lat')
    lng_raw = data.get('lng')

    if not name:
        return jsonify({"error": "invalid_input", "message": "NGO name is required"}), 400
    if not phone:
        return jsonify({"error": "invalid_input", "message": "Phone number is required"}), 400
    if not address:
        return jsonify({"error": "invalid_input", "message": "Address is required"}), 400

    if lat_raw is None or lng_raw is None:
        return jsonify({"error": "invalid_input", "message": "Latitude and Longitude are required"}), 400

    try:
        lat = float(lat_raw)
        lng = float(lng_raw)
    except (ValueError, TypeError):
        return jsonify({"error": "invalid_input", "message": "Latitude and Longitude must be valid numbers"}), 400

    if math.isnan(lat) or math.isinf(lat) or not (-90.0 <= lat <= 90.0):
        return jsonify({"error": "invalid_input", "message": "Latitude must be between -90 and 90"}), 400

    if math.isnan(lng) or math.isinf(lng) or not (-180.0 <= lng <= 180.0):
        return jsonify({"error": "invalid_input", "message": "Longitude must be between -180 and 180"}), 400

    new_ngo = NGO(
        name=name,
        phone=phone,
        address=address,
        lat=lat,
        lng=lng
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
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required to notify NGOs."}), 403

    user_id = admin.id
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