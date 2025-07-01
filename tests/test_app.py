import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application import app

def test_app():
    response = app.test_client().get("/")

    assert response.status_code == 200
    # Update the assertion to match what your app actually returns
    assert b"Student Exam Performance Indicator" in response.data