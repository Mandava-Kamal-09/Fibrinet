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

    def parse_request(self, request_str: str):
        """
        Parses the export request string and returns corresponding 
        export strategies and folder location.

        Args:
            request_str (str): The request string in the format 
                               'export_request <data_strategy> <image_strategy> <folder_location>'.

        Returns:
            dict: A dictionary with data export strategy, image export strategy, and folder location.

        Raises:
            ValueError: If the request string is invalid or missing parts.
        """
        Logger.log(f"start parse_request(self, {request_str})")
        Logger.log(f"Parsing request: {request_str}")
        
        # REMOVE ANY LEADING/TRAILING SPACES
        request_str = request_str.strip()  
        
        # CHECK IF REQUEST STARTS WITH 'EXPORT_REQUEST'
        if not request_str.startswith("export_request"):
            Logger.log("Request does not start with 'export_request'")
            raise ValueError("The request must start with 'export_request'")
        
        Logger.log("Request starts correctly with 'export_request', stripping prefix")
        # REMOVE 'EXPORT_REQUEST' PREFIX
        request_str = request_str[len("export_request"):].strip()  
        
        # SPLIT THE REMAINING STRING INTO PARTS
        parts = request_str.split(maxsplit=2)  # LIMIT TO 3 PARTS
        # LOG THE SPLIT PARTS
        Logger.log(f"Split request into parts: {parts}")

        # ENSURE THERE ARE EXACTLY 3 PARTS
        if len(parts) != 3:
            Logger.log(f"Invalid number of parts in the request, expected 3 but got {len(parts)}")
            raise ValueError("Request must consist of exactly 3 parts: data, image, folder.")

        # ASSIGN VALUES OR SET TO NONE IF 'NONE' IS PROVIDED
        data_export_strategy = parts[0] if parts[0].lower() != "none" else None
        image_export_strategy = parts[1] if parts[1].lower() != "none" else None
        folder_location = parts[2] if parts[2].lower() != "none" else None

        Logger.log(f"Extracted data_export_strategy: {data_export_strategy}, image_export_strategy: {image_export_strategy}, folder_location: {folder_location}")

        # CHECK IF AT LEAST ONE STRATEGY IS PROVIDED
        if data_export_strategy is None and image_export_strategy is None:
            Logger.log("Neither data nor image export strategy provided.")
            raise ValueError("At least one of 'data_export_strategy' or 'image_export_strategy' must be provided.")

        # ENSURE A VALID FILE LOCATION IS PROVIDED
        if not folder_location:
            Logger.log("File location is missing.")
            raise ValueError("File location must be provided and cannot be empty.")
        elif folder_location.lower() == "none":
            Logger.log("File location is 'none', which is not allowed.")
            raise ValueError("File location cannot be 'none'.")

        # INSTANTIATE DATA EXPORT STRATEGY IF VALID
        if data_export_strategy:
            if data_export_strategy not in self.VALID_DATA_STRATEGIES:
                Logger.log(f"Invalid data export strategy: {data_export_strategy}")
                raise ValueError(f"Invalid data export strategy: '{data_export_strategy}'.")
            Logger.log(f"Instantiating data export strategy: {data_export_strategy}")
            # INSTANTIATE THE STRATEGY
            data_export_strategy = self.VALID_DATA_STRATEGIES[data_export_strategy]() 

        # INSTANTIATE IMAGE EXPORT STRATEGY IF VALID
        if image_export_strategy:
            if image_export_strategy not in self.VALID_IMAGE_STRATEGIES:
                Logger.log(f"Invalid image export strategy: {image_export_strategy}")
                raise ValueError(f"Invalid image export strategy: '{image_export_strategy}'.")
            Logger.log(f"Instantiating image export strategy: {image_export_strategy}")
            # INSTANTIATE THE STRATEGY
            image_export_strategy = self.VALID_IMAGE_STRATEGIES[image_export_strategy]()  
        
        Logger.log("Request parsed successfully. Returning result.")
        Logger.log("start parse_request(self, request_str)")

        # RETURN THE RESULTS AS A DICTIONARY
        return {
            'data_export_strategy': data_export_strategy,
            'image_export_strategy': image_export_strategy,
            'folder_location': folder_location
        }
