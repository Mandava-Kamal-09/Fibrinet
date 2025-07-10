
import unittest
import os
import sys
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_main import app, controller
from src.controllers.system_controller import SystemController
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_main import app
import pandas as pd
import io

class TestApi(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.test_file_path = "test_network.xlsx"
        self.create_test_excel_file()
        # Access the global controller from api_main and reset its state
        controller.network_manager.reset_network_and_state()
        # Load a network for tests that require it
        with open(self.test_file_path, "rb") as f:
            file_content = f.read()
            controller.input_network(io.BytesIO(file_content))

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def create_test_excel_file(self):
        nodes_data = {'n_id': [1, 2], 'n_x': [0, 1], 'n_y': [0, 1]}
        edges_data = {'e_id': [1], 'n_from': [1], 'n_to': [2]}
        meta_data = {'key': ['spring_stiffness_constant'], 'value': [1.0]}

        with pd.ExcelWriter(self.test_file_path) as writer:
            pd.DataFrame(nodes_data).to_excel(writer, sheet_name='Sheet1', index=False)
            pd.DataFrame(edges_data).to_excel(writer, sheet_name='Sheet1', index=False, startrow=4)
            pd.DataFrame(meta_data).to_excel(writer, sheet_name='Sheet1', index=False, startrow=7)

    def test_read_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "FibriNet API is running!"})

    def test_input_network(self):
        with open(self.test_file_path, "rb") as f:
            response = self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Network loaded successfully from test_network.xlsx"})

    def test_degrade_node(self):
        with open(self.test_file_path, "rb") as f:
            self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        response = self.client.post("/degrade_node/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Node 1 degraded successfully"})

    def test_degrade_edge(self):
        with open(self.test_file_path, "rb") as f:
            self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        response = self.client.post("/degrade_edge/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Edge 1 degraded successfully"})

    def test_undo_degradation(self):
        with open(self.test_file_path, "rb") as f:
            self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        self.client.post("/degrade_node/1")
        response = self.client.post("/undo_degradation/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Last degradation undone successfully"})

    def test_redo_degradation(self):
        with open(self.test_file_path, "rb") as f:
            self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        self.client.post("/degrade_node/1")
        self.client.post("/undo_degradation/")
        response = self.client.post("/redo_degradation/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Last undone degradation redone successfully"})

    def test_run_simulation(self):
        with open(self.test_file_path, "rb") as f:
            response = self.client.post("/run_simulation/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        self.assertEqual(response.status_code, 200)
        self.assertIn("attachment; filename=\"simulation_results_test_network.xlsx.csv\"", response.headers["content-disposition"])

    def test_analyze_network(self):
        with open(self.test_file_path, "rb") as f:
            self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        response = self.client.get("/analyze_network/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        self.assertIn("results", response.json())

    def test_export_data(self):
        with open(self.test_file_path, "rb") as f:
            self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        response = self.client.post("/export_data/", json={"data_format": "excel", "image_format": "png", "path": "."})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Data exported successfully"})

    def test_update_parameters(self):
        with open(self.test_file_path, "rb") as f:
            self.client.post("/input_network/", files={"file": ("test_network.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
        response = self.client.post("/update_parameters/", json={"params": {"key": "value"}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Parameters updated successfully"})

if __name__ == '__main__':
    unittest.main()
