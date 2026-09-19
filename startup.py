import os

from models import Case, User
from services.geocoding import resolve_coordinates_to_address


def run_startup_migrations(db):
    """Ensure newly added columns exist in the case table (SQLite + Postgres),
    then migrate any legacy lat/long values stored in case.location into
    readable wording addresses."""
    try:
        with db.engine.connect() as conn:
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN reported_by_id INTEGER REFERENCES "user"(id)'))
                conn.commit()
                print("[startup] Added missing reported_by_id column to case table.")
            except Exception:
                conn.rollback()
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN reviewed_by_id INTEGER REFERENCES "user"(id)'))
                conn.commit()
            except Exception:
                conn.rollback()
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN vet_confirmed_label VARCHAR(100)'))
                conn.commit()
            except Exception:
                conn.rollback()
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN image_hash VARCHAR(64)'))
                conn.commit()
                print("[startup] Added missing image_hash column to case table.")
            except Exception:
                conn.rollback()
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN latitude FLOAT'))
                conn.commit()
                print("[startup] Added missing latitude column to case table.")
            except Exception:
                conn.rollback()
            try:
                conn.execute(db.text('ALTER TABLE "case" ADD COLUMN longitude FLOAT'))
                conn.commit()
                print("[startup] Added missing longitude column to case table.")
            except Exception:
                conn.rollback()
            # prediction/confidence must allow NULL — predict_image() returns
            # prediction=None (and confidence=None for OOD cases) for the
            # "not_recognized" and "unable_to_classify" statuses (ML v2 Step
            # 10). Existing tables created before this change still have the
            # old NOT NULL constraint, so it must be dropped explicitly;
            # db.create_all() never alters existing tables. Postgres-only
            # syntax (matches DATABASE_URL in production); harmlessly no-ops
            # on SQLite via the except below.
            try:
                conn.execute(db.text('ALTER TABLE "case" ALTER COLUMN prediction DROP NOT NULL'))
                conn.commit()
                print("[startup] Made case.prediction nullable.")
            except Exception:
                conn.rollback()
            try:
                conn.execute(db.text('ALTER TABLE "case" ALTER COLUMN confidence DROP NOT NULL'))
                conn.commit()
                print("[startup] Made case.confidence nullable.")
            except Exception:
                conn.rollback()

            # Data migration: convert any lat/long coordinates stored in
            # case.location into readable wording addresses.
            try:
                raw_cases = Case.query.filter(Case.location.isnot(None)).all()
                migrated_count = 0
                for c in raw_cases:
                    if c.location:
                        parts = c.location.split(',')
                        if len(parts) == 2:
                            try:
                                p_lat = float(parts[0].strip())
                                p_lng = float(parts[1].strip())
                                if -90.0 <= p_lat <= 90.0 and -180.0 <= p_lng <= 180.0:
                                    if c.latitude is None:
                                        c.latitude = p_lat
                                    if c.longitude is None:
                                        c.longitude = p_lng
                                    wording = resolve_coordinates_to_address(p_lat, p_lng)
                                    if wording:
                                        c.location = wording
                                        migrated_count += 1
                            except (ValueError, TypeError):
                                conn.rollback()
                if migrated_count > 0:
                    db.session.commit()
                    print(f"[startup] Migrated {migrated_count} case(s) from lat/long to wording addresses.")
            except Exception as mig_data_err:
                print(f"[startup] Location wording migration notice: {mig_data_err}")
    except Exception as mig_err:
        print(f"[startup] Migration notice: {mig_err}")


def seed_fixed_admin(db):
    """Create or sync the permanent admin account from environment config."""
    FIXED_ADMIN_USERNAME = os.environ.get('FIXED_ADMIN_USERNAME', 'admin')
    FIXED_ADMIN_PASSWORD = os.environ.get('FIXED_ADMIN_PASSWORD', 'admin123')
    FIXED_ADMIN_EMAIL = os.environ.get('FIXED_ADMIN_EMAIL', 'admin@pawcare.local')

    if not (FIXED_ADMIN_USERNAME and FIXED_ADMIN_PASSWORD):
        print("[startup] FIXED_ADMIN_USERNAME / FIXED_ADMIN_PASSWORD not set — skipping admin auto-seed.")
        return

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
