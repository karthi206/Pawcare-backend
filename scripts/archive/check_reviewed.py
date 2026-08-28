# check_reviewed.py
from app import app, db
from models import Case

with app.app_context():
    reviewed = Case.query.filter(Case.vet_confirmed_label.isnot(None)).all()
    for c in reviewed:
        print(f"id={c.id}, filename={c.filename}, vet_label={c.vet_confirmed_label}")