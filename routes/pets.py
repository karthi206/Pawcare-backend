from datetime import datetime

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import cloudinary.uploader

from extensions import db
from models import Pet, AdoptionRequest
from utils.validation import validate_image_file
from helpers import require_admin

pets_bp = Blueprint('pets', __name__)


@pets_bp.route('/pets', methods=['GET'])
@pets_bp.route('/api/pets', methods=['GET'])
def get_pets():
    pets = Pet.query.filter_by(status='available').all()
    return jsonify([p.to_dict() for p in pets])


@pets_bp.route('/pets/<int:pet_id>/adopt', methods=['POST'])
@pets_bp.route('/api/pets/<int:pet_id>/adopt', methods=['POST'])
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


@pets_bp.route('/pets', methods=['POST'])
@pets_bp.route('/api/pets', methods=['POST'])
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
