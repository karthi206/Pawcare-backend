from app import app, db
from models import Case

with app.app_context():
    cases = Case.query.all()
    for case in cases:
        print(case.to_dict())