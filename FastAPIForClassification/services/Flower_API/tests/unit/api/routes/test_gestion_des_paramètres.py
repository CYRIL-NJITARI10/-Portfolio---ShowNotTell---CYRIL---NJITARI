def test_get_parameters(test_client):
    response = test_client.get("/get-parameters")
    assert response.status_code == 200
    assert "n_estimators" in response.json()
    assert "criterion" in response.json()


def test_update_parameters(test_client):
    new_params = {"n_estimators": 50, "criterion": "entropy"}
    response = test_client.put("/update-parameters", json=new_params)
    assert response.status_code == 200
    assert "message" in response.json()

    # Vous pouvez également vérifier que les paramètres ont été mis à jour correctement en appelant /get-parameters ici et en vérifiant les valeurs.

def test_add_parameters(test_client):
    new_params = {"n_estimators": 50, "criterion": "entropy"}
    response = test_client.post("/add-parameters", json=new_params)
    assert response.status_code == 200
    assert "message" in response.json()

    # Vous pouvez également vérifier que les nouveaux paramètres ont été ajoutés correctement en appelant /get-parameters ici et en vérifiant les valeurs.

