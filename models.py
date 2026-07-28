from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Case(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    is_uncertain = db.Column(db.Boolean, default=False)
    status = db.Column(db.String(50), default="pending")
    location = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    vet_confirmed_label = db.Column(db.String(100), nullable=True)  # new
    # In models.py, add to Case:
    reviewed_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "filename": self.filename,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "is_uncertain": self.is_uncertain,
            "status": self.status,
            "location": self.location,
            "created_at": self.created_at.isoformat(),
            "vet_confirmed_label": self.vet_confirmed_label  # new
        }

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default="user")  # 'user', 'vet', or 'admin'

    # Vet-specific fields - only relevant when role='vet'
    license_number = db.Column(db.String(100), nullable=True)
    clinic_name = db.Column(db.String(200), nullable=True)
    clinic_address = db.Column(db.String(300), nullable=True)
    is_verified = db.Column(db.Boolean, default=False)  # admin must approve vets before they're trusted

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "license_number": self.license_number,
            "clinic_name": self.clinic_name,
            "clinic_address": self.clinic_address,
            "is_verified": self.is_verified,
        }