def test_create_ngo_duplicate_returns_409(client, make_user):
    admin = make_user(username='adminuser', role='admin')
    login_resp = client.post('/auth/login', json={"username": "adminuser", "password": "pass1234"})
    assert login_resp.status_code == 200
    csrf_token = login_resp.get_json()['csrf_token']
    headers = {'X-CSRF-TOKEN': csrf_token}

    ngo_data = {
        "name": "Happy Paws Rescue",
        "phone": "+1234567890",
        "email": "contact@happypaws.org",
        "address": "123 Animal Lane",
        "lat": 12.9716,
        "lng": 77.5946
    }

    # First creation should succeed
    resp1 = client.post('/ngos', json=ngo_data, headers=headers)
    assert resp1.status_code == 201

    # Second creation with same name & address (with extra whitespace to test stripping) should fail with 409
    dup_data = dict(ngo_data)
    dup_data["name"] = "  Happy Paws Rescue  "
    dup_data["address"] = "  123 Animal Lane  "
    resp2 = client.post('/ngos', json=dup_data, headers=headers)
    assert resp2.status_code == 409
    body = resp2.get_json()
    assert body.get('error') == 'conflict'
    assert 'already exists' in body.get('message', '').lower()
