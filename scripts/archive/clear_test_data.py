# clear_test_data.py
from app import app, db
from models import Case

with app.app_context():
    # Delete only the fake seed entries (filenames test1.jpg through test7.jpg)
    fake_cases = Case.query.filter(Case.filename.like('test%.jpg')).all()
    for case in fake_cases:
        db.session.delete(case)
    db.session.commit()
    print(f"Deleted {len(fake_cases)} test cases.")