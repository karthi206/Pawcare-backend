def test_register_requires_all_fields(client):
    resp = client.post('/auth/register', json={"username": "alice"})
    assert resp.status_code == 400


def test_register_rejects_short_password(client):
    resp = client.post('/auth/register', json={
        "username": "alice",
        "email": "alice@example.com",
        "password": "a",
    })
    assert resp.status_code == 400
    assert 'password' in resp.get_json().get('error', '').lower() or \
           'password' in resp.get_json().get('message', '').lower()


def test_register_accepts_valid_password(client):
    resp = client.post('/auth/register', json={
        "username": "alice",
        "email": "alice@example.com",
        "password": "pass1234",
    })
    assert resp.status_code == 201
    body = resp.get_json()
    assert body['user']['username'] == 'alice'


def test_register_rejects_duplicate_username(client, make_user):
    make_user(username='alice', email='existing@example.com')
    resp = client.post('/auth/register', json={
        "username": "alice",
        "email": "new@example.com",
        "password": "pass1234",
    })
    assert resp.status_code == 409


def test_register_rejects_invalid_role(client):
    resp = client.post('/auth/register', json={
        "username": "alice",
        "email": "alice@example.com",
        "password": "pass1234",
        "role": "superadmin",
    })
    assert resp.status_code == 400


def test_login_wrong_password_fails(client, make_user):
    make_user(username='bob', password='correctpass')
    resp = client.post('/auth/login', json={
        "username": "bob",
        "password": "wrongpass",
    })
    assert resp.status_code == 401


def test_login_unknown_user_fails(client):
    resp = client.post('/auth/login', json={
        "username": "doesnotexist",
        "password": "whatever123",
    })
    assert resp.status_code == 401


def test_login_success_sets_cookie_and_csrf(client, make_user):
    make_user(username='bob', password='correctpass')
    resp = client.post('/auth/login', json={
        "username": "bob",
        "password": "correctpass",
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body['user']['username'] == 'bob'
    assert body.get('csrf_token') is not None
    # httpOnly JWT cookie should be set
    cookie_names = [c.name for c in client.get_cookies()] if hasattr(client, 'get_cookies') else \
                   [c for c in resp.headers.get_all('Set-Cookie')]
    assert any('access_token' in str(c) for c in cookie_names)


def test_login_rate_limited_after_five_attempts(client, make_user):
    make_user(username='bob', password='correctpass')
    last_status = None
    for _ in range(6):
        resp = client.post('/auth/login', json={
            "username": "bob",
            "password": "wrongpass",
        })
        last_status = resp.status_code
    assert last_status == 429


def test_me_requires_auth(client):
    resp = client.get('/auth/me')
    assert resp.status_code == 401


def test_me_returns_user_after_login(client, make_user):
    make_user(username='bob', password='correctpass')
    login_resp = client.post('/auth/login', json={
        "username": "bob",
        "password": "correctpass",
    })
    assert login_resp.status_code == 200

    me_resp = client.get('/auth/me')
    assert me_resp.status_code == 200
    assert me_resp.get_json()['username'] == 'bob'