from .excel_export_strategy import ExcelExportStrategy
from .png_export_strategy import PngExportStrategy
from utils.logger.logger import Logger

class ExportRequestInterpreter:
    """Parse export requests and instantiate strategies."""

    VALID_DATA_STRATEGIES = {
        "excel_data_export_strategy": ExcelExportStrategy
    }

    VALID_IMAGE_STRATEGIES = {
        "png_image_export_strategy": PngExportStrategy
    }

    def parse_request(self, request_str: str):
        """Return dict with strategies and folder from a request string."""
        Logger.log(f"start parse_request(self, {request_str})")
        Logger.log(f"Parsing request: {request_str}")
        request_str = request_str.strip()
        if not request_str.startswith("export_request"):
            Logger.log("Request does not start with 'export_request'")
            raise ValueError("The request must start with 'export_request'")
        Logger.log("Stripping prefix and splitting parts")
        request_str = request_str[len("export_request"):].strip()
        parts = request_str.split(maxsplit=2)
        Logger.log(f"Split request into parts: {parts}")

        if len(parts) != 3:
            Logger.log(f"Invalid number of parts in the request, expected 3 but got {len(parts)}")
            raise ValueError("Request must consist of exactly 3 parts: data, image, folder.")

        data_export_strategy = parts[0] if parts[0].lower() != "none" else None
        image_export_strategy = parts[1] if parts[1].lower() != "none" else None
        folder_location = parts[2] if parts[2].lower() != "none" else None

        Logger.log(f"Extracted data_export_strategy: {data_export_strategy}, image_export_strategy: {image_export_strategy}, folder_location: {folder_location}")

        if data_export_strategy is None and image_export_strategy is None:
            Logger.log("Neither data nor image export strategy provided.")
            raise ValueError("At least one of 'data_export_strategy' or 'image_export_strategy' must be provided.")

        if not folder_location:
            Logger.log("File location is missing.")
            raise ValueError("File location must be provided and cannot be empty.")
        elif folder_location.lower() == "none":
            Logger.log("File location is 'none', which is not allowed.")
            raise ValueError("File location cannot be 'none'.")

        if data_export_strategy:
            if data_export_strategy not in self.VALID_DATA_STRATEGIES:
                Logger.log(f"Invalid data export strategy: {data_export_strategy}")
                raise ValueError(f"Invalid data export strategy: '{data_export_strategy}'.")
            Logger.log(f"Instantiating data export strategy: {data_export_strategy}")
            data_export_strategy = self.VALID_DATA_STRATEGIES[data_export_strategy]() 

        if image_export_strategy:
            if image_export_strategy not in self.VALID_IMAGE_STRATEGIES:
                Logger.log(f"Invalid image export strategy: {image_export_strategy}")
                raise ValueError(f"Invalid image export strategy: '{image_export_strategy}'.")
            Logger.log(f"Instantiating image export strategy: {image_export_strategy}")
            image_export_strategy = self.VALID_IMAGE_STRATEGIES[image_export_strategy]()  
        
        Logger.log("Request parsed successfully.")

        return {
            'data_export_strategy': data_export_strategy,
            'image_export_strategy': image_export_strategy,
            'folder_location': folder_location
        }
