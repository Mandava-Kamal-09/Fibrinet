import os
from datetime import datetime
from .export_request_interpreter import ExportRequestInterpreter
from utils.logger.logger import Logger

class ExportManager:
    def __init__(self):
        self.interpreter = ExportRequestInterpreter()

    def handle_export_request(self, network_state_history, export_request):
        """Parse request, run strategies, and save outputs."""
        try: 
            export_request_data = self.interpreter.parse_request(export_request)
            data_export_strategy = export_request_data.get('data_export_strategy', None)
            image_export_strategy = export_request_data.get('image_export_strategy', None)
            base_folder_location = export_request_data.get('folder_location', None)

            self._verify_folder(base_folder_location)

            # Generate exports
            data_export = None
            image_export = None

            if data_export_strategy:
                data_export = data_export_strategy.generate_export(network_state_history)
            
            if image_export_strategy:
                image_export = image_export_strategy.generate_export(network_state_history)

            # Create folder structure and save
            self._save_reports(data_export, image_export, base_folder_location)

        except Exception as ex:
            Logger.log(f"Error handling export request: {str(ex)}")
            raise

    def _create_export_folders(self, base_folder_location):
        """Create timestamped root folder for this export."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_folder = os.path.join(base_folder_location, f"export_{timestamp}")
        
        # Create root folder
        if not os.path.exists(root_folder):
            try:
                os.makedirs(root_folder)
                Logger.log(f"Created root folder: {root_folder}")
            except OSError as e:
                Logger.log(f"Error creating root folder {root_folder}: {e}")
                raise
        
        # Return the root folder path
        return root_folder

    def _save_reports(self, data_export, image_export, base_folder_location):
        """Save generated files into data/image subfolders."""
        # Create root folder
        root_folder = self._create_export_folders(base_folder_location)

        # Create subfolders only if there is content
        if data_export:
            data_export_folder = os.path.join(root_folder, 'data_export')
            self._create_and_save_files(data_export, data_export_folder)

        if image_export:
            image_export_folder = os.path.join(root_folder, 'image_export')
            self._create_and_save_files(image_export, image_export_folder)

    def _create_and_save_files(self, files, folder_location):
        """Create folder and write files."""
        # Create folder
        if not os.path.exists(folder_location):
            try:
                os.makedirs(folder_location)
                Logger.log(f"Created folder: {folder_location}")
            except OSError as e:
                Logger.log(f"Error creating folder {folder_location}: {e}")
                raise
        
        # Save files
        self._save_files(files, folder_location)

    def _save_files(self, files, folder_location):
        """Write each (filename, bytes) to disk."""
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
        """Ensure base folder exists and is a directory."""
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
