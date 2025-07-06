from .excel_export_strategy import ExcelExportStrategy
from .png_export_strategy import PngExportStrategy
from .json_export_strategy import JsonExportStrategy
from .csv_export_strategy import CsvExportStrategy
from utils.logger.logger import Logger

class ExportRequestInterpreter:
    """
    This class interprets export requests and instantiates the
    appropriate export strategies based on the request.
    """

    # VALID DATA EXPORT STRATEGIES
    VALID_DATA_STRATEGIES = {
        "excel": ExcelExportStrategy,
        "json": JsonExportStrategy,
        "csv": CsvExportStrategy
    }

    # VALID IMAGE EXPORT STRATEGIES
    VALID_IMAGE_STRATEGIES = {
        "png": PngExportStrategy
    }

    def parse_request(self, request: dict):
        """
        Parses the export request dictionary and returns corresponding 
        export strategies and folder location.

        Args:
            request (dict): The request dictionary containing 'data_format', 'image_format', and 'path'.

        Returns:
            dict: A dictionary with data export strategy, image export strategy, and folder location.

        Raises:
            ValueError: If the request dictionary is invalid or missing parts.
        """
        Logger.log(f"start parse_request(self, {request})")
        Logger.log(f"Parsing request: {request}")
        
        data_format = request.get('data_format')
        image_format = request.get('image_format')
        folder_location = request.get('path')

        # ASSIGN VALUES OR SET TO NONE IF 'NONE' IS PROVIDED
        data_export_strategy_name = data_format if data_format and data_format.lower() != "none" else None
        image_export_strategy_name = image_format if image_format and image_format.lower() != "none" else None

        Logger.log(f"Extracted data_export_strategy: {data_export_strategy_name}, image_export_strategy: {image_export_strategy_name}, folder_location: {folder_location}")

        # CHECK IF AT LEAST ONE STRATEGY IS PROVIDED
        if data_export_strategy_name is None and image_export_strategy_name is None:
            Logger.log("Neither data nor image export strategy provided.")
            raise ValueError("At least one of 'data_export_strategy' or 'image_export_strategy' must be provided.")

        # ENSURE A VALID FILE LOCATION IS PROVIDED
        if not folder_location:
            Logger.log("File location is missing.")
            raise ValueError("File location must be provided and cannot be empty.")
        elif folder_location.lower() == "none":
            Logger.log("File location is 'none', which is not allowed.")
            raise ValueError("File location cannot be 'none'.")

        data_export_strategy = None
        if data_export_strategy_name:
            if data_export_strategy_name not in self.VALID_DATA_STRATEGIES:
                Logger.log(f"Invalid data export strategy: {data_export_strategy_name}")
                raise ValueError(f"Invalid data export strategy: '{data_export_strategy_name}'.")
            Logger.log(f"Instantiating data export strategy: {data_export_strategy_name}")
            data_export_strategy = self.VALID_DATA_STRATEGIES[data_export_strategy_name]() 

        image_export_strategy = None
        if image_export_strategy_name:
            if image_export_strategy_name not in self.VALID_IMAGE_STRATEGIES:
                Logger.log(f"Invalid image export strategy: {image_export_strategy_name}")
                raise ValueError(f"Invalid image export strategy: '{image_export_strategy_name}'.")
            Logger.log(f"Instantiating image export strategy: {image_export_strategy_name}")
            image_export_strategy = self.VALID_IMAGE_STRATEGIES[image_export_strategy_name]()  
        
        Logger.log("Request parsed successfully. Returning result.")

        # RETURN THE RESULTS AS A DICTIONARY
        return {
            'data_export_strategy': data_export_strategy,
            'image_export_strategy': image_export_strategy,
            'folder_location': folder_location
        }
