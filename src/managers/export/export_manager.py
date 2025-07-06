import os
from datetime import datetime
from .export_request_interpreter import ExportRequestInterpreter
from utils.logger.logger import Logger

class ExportManager:
    def __init__(self):
        self.interpreter = ExportRequestInterpreter()

    def handle_export_request(self, network_state_history, export_request):
        try: 
            export_request_data = self.interpreter.parse_request(export_request)
            data_export_strategy = export_request_data.get('data_export_strategy', None)
            image_export_strategy = export_request_data.get('image_export_strategy', None)
            base_folder_location = export_request_data.get('folder_location', None)

            self._verify_folder(base_folder_location)

            # GENERATE EXPORTS
            data_export = None
            image_export = None

            if data_export_strategy:
                data_export = data_export_strategy.generate_export(network_state_history)
            
            if image_export_strategy:
                image_export = image_export_strategy.generate_export(network_state_history)

            # CREATE FOLDER STRUCTURE AND SAVE REPORTS
            root_folder = self._create_export_folders(base_folder_location)
            self._save_reports(data_export, image_export, root_folder)

        except Exception as ex:
            Logger.log(f"Error handling export request: {str(ex)}")
            raise

    def _create_export_folders(self, base_folder_location):
        """Create root folder and subfolders based on export type, only if needed."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_folder = os.path.join(base_folder_location, f"export_{timestamp}")
        
        # Create root folder if it doesn't exist
        if not os.path.exists(root_folder):
            try:
                os.makedirs(root_folder)
                Logger.log(f"Created root folder: {root_folder}")
            except OSError as e:
                Logger.log(f"Error creating root folder {root_folder}: {e}")
                raise
        
        # Return the root folder path
        return root_folder

    def _save_reports(self, data_export, image_export, root_folder):

        # Create subfolders for data and image exports only if there is content
        if data_export:
            data_export_folder = os.path.join(root_folder, 'data_export')
            self._create_and_save_files(data_export, data_export_folder)

        if image_export:
            image_export_folder = os.path.join(root_folder, 'image_export')
            self._create_and_save_files(image_export, image_export_folder)

    def _create_and_save_files(self, files, folder_location):
        """Helper function to create the folder and save files to it."""
        # Create folder if it doesn't exist
        if not os.path.exists(folder_location):
            try:
                os.makedirs(folder_location)
                Logger.log(f"Created folder: {folder_location}")
            except OSError as e:
                Logger.log(f"Error creating folder {folder_location}: {e}")
                raise
        
        # Save the files to the folder
        self._save_files(files, folder_location)

    def _save_files(self, files, folder_location):
        """Helper function to save a list of files to a folder."""
        if not files:
            return

        for filename, content in files:
            file_path = os.path.join(folder_location, filename)
            try:
                with open(file_path, 'wb') as f: 
                    f.write(content)
                Logger.log(f"Saved file: {file_path}")
            except Exception as e:
                Logger.log(f"Error saving file {file_path}: {e}")
                raise

    def _verify_folder(self, folder_path):
        """Verifies if the folder exists and creates it if it doesn't."""
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                Logger.log(f"Created folder: {folder_path}")
            except OSError as e:
                Logger.log(f"Error creating folder {folder_path}: {e}")
                raise 
        elif not os.path.isdir(folder_path):
            Logger.log(f"{folder_path} exists but is not a directory.")
            raise ValueError(f"{folder_path} exists but is not a directory.")
        Logger.log(f"Folder verified: {folder_path}")
