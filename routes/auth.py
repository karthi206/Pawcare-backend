from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, jwt_required, get_jwt_identity,
    set_access_cookies, unset_jwt_cookies, get_csrf_token, get_jwt,
)

from extensions import db
from models import User
from email_service import send_vet_registration_email

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/auth/register', methods=['POST'])
@auth_bp.route('/api/auth/register', methods=['POST'])
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

    if new_user.role == "vet":
        send_vet_registration_email(new_user)

    return jsonify({
        "message": "Registered successfully" if role == 'user' else "Registered — awaiting admin verification before you can review cases",
        "user": new_user.to_dict()
    }), 201


@auth_bp.route('/auth/login', methods=['POST'])
@auth_bp.route('/api/auth/login', methods=['POST'])
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


@auth_bp.route('/auth/logout', methods=['POST'])
@auth_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    resp = jsonify({"message": "Logged out"})
    unset_jwt_cookies(resp)
    return resp


@auth_bp.route('/auth/me', methods=['GET'])
@auth_bp.route('/api/auth/me', methods=['GET'])
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


@auth_bp.route('/auth/profile', methods=['PATCH'])
@auth_bp.route('/api/auth/profile', methods=['PATCH'])
@jwt_required()
def update_profile():
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

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid_input", "message": "Request body must be a JSON object"}), 400

    # Privilege escalation protection: prevent modifying role, id, or is_verified
    if 'role' in data and data['role'] != user.role:
        return jsonify({"error": "Role modification is not permitted."}), 400
    if 'is_verified' in data and data['is_verified'] != user.is_verified:
        return jsonify({"error": "Verification status cannot be modified."}), 400
    if 'id' in data and data['id'] != user.id:
        return jsonify({"error": "User ID cannot be modified."}), 400

    if 'username' in data:
        new_username = (data.get('username') or '').strip()
        if not new_username:
            return jsonify({"error": "Username cannot be empty"}), 400
        if len(new_username) < 3 or len(new_username) > 80:
            return jsonify({"error": "Username must be between 3 and 80 characters"}), 400
        existing = User.query.filter(User.username == new_username, User.id != user.id).first()
        if existing:
            return jsonify({"error": "Username is already taken"}), 409
        user.username = new_username

    if 'email' in data:
        new_email = (data.get('email') or '').strip().lower()
        if not new_email:
            return jsonify({"error": "Email cannot be empty"}), 400
        if '@' not in new_email or '.' not in new_email:
            return jsonify({"error": "A valid email is required"}), 400
        existing = User.query.filter(User.email == new_email, User.id != user.id).first()
        if existing:
            return jsonify({"error": "Email is already taken"}), 409
        user.email = new_email

    # Vet-specific fields - only updated if user is a vet, ignored if not a vet
    if user.role == 'vet':
        if 'license_number' in data:
            val = str(data['license_number']).strip() if data['license_number'] is not None else None
            user.license_number = val if val else None
        if 'clinic_name' in data:
            val = str(data['clinic_name']).strip() if data['clinic_name'] is not None else None
            user.clinic_name = val if val else None
        if 'clinic_address' in data:
            val = str(data['clinic_address']).strip() if data['clinic_address'] is not None else None
            user.clinic_address = val if val else None

    db.session.commit()
    return jsonify({
        "message": "Profile updated successfully",
        "user": user.to_dict()
    }), 200


@auth_bp.route('/auth/password', methods=['PATCH'])
@auth_bp.route('/api/auth/password', methods=['PATCH'])
@jwt_required()
def update_password():
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

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "invalid_input", "message": "Request body must be a JSON object"}), 400

    current_password = data.get('current_password')
    new_password = data.get('new_password')

    if not current_password or not new_password:
        return jsonify({"error": "Current password and new password are required"}), 400

    if not user.check_password(current_password):
        return jsonify({"error": "Current password is incorrect"}), 400

    if not isinstance(new_password, str) or len(new_password.strip()) < 6:
        return jsonify({"error": "New password must be at least 6 characters long"}), 400

    user.set_password(new_password)
    db.session.commit()

    return jsonify({"message": "Password updated successfully"}), 200
