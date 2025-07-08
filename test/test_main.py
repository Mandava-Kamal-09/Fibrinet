
import unittest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from src.controllers.system_controller import SystemController
from src.models.exceptions import StateTransitionError, InvalidInputDataError, UnsupportedFileTypeError, NodeNotFoundError, EdgeNotFoundError
from src.managers.network.networks.base_network import BaseNetwork

class TestFibrinet(unittest.TestCase):

    def setUp(self):
        self.controller = SystemController()
        self.test_file_path = "test_network.xlsx"
        self.create_test_excel_file()

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

    def test_input_network(self):
        self.controller.input_network(self.test_file_path)
        self.assertIsNotNone(self.controller.network_manager.get_network())
        self.assertTrue(self.controller.system_state.network_loaded)

    def test_degrade_node(self):
        self.controller.input_network(self.test_file_path)
        self.controller.degrade_node(1)
        self.assertIsNone(self.controller.network_manager.get_network().get_node_by_id(1))

    def test_degrade_edge(self):
        self.controller.input_network(self.test_file_path)
        self.controller.degrade_edge(1)
        self.assertIsNone(self.controller.network_manager.get_network().get_edge_by_id(1))

    def test_undo_redo_degradation(self):
        self.controller.input_network(self.test_file_path)
        initial_network = self.controller.network_manager.get_network()
        
        self.controller.degrade_node(1)
        degraded_network = self.controller.network_manager.get_network()
        self.assertNotEqual(initial_network, degraded_network)

        self.controller.undo_degradation()
        undone_network = self.controller.network_manager.get_network()
        self.assertEqual(initial_network.get_nodes()[0].get_id(), undone_network.get_nodes()[0].get_id())

        self.controller.redo_degradation()
        redone_network = self.controller.network_manager.get_network()
        self.assertEqual(degraded_network.get_nodes()[0].get_id(), redone_network.get_nodes()[0].get_id())

if __name__ == '__main__':
    unittest.main()
