"""
Shared Flask extension instances.

Kept separate from app.py so route/service modules can import `db` and `jwt`
without triggering a circular import with the app factory.
"""
from models import db
from flask_jwt_extended import JWTManager

jwt = JWTManager()
