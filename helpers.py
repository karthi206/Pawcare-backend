from flask_jwt_extended import get_jwt_identity

from extensions import db
from models import User


def get_current_user_obj():
    """Helper: load the User row for the current JWT identity safely, or None."""
    try:
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
