from utils.logger.logger import Logger

class BaseNode:
    """Node with attributes and schema validation."""

    schema = {"n_id": int}

    def __init__(self, attributes):
        """Initialize with attributes dict and validate against schema."""
        Logger.log(f"start Node __init__(self)")

        self.attributes = []

        for key, value in attributes.items():
            setattr(self, key, value)
            self.attributes.append(key)
            Logger.log(f"Node attribute added {key}={value}")

        if not self.validate_attributes():
            raise ValueError("Invalid node attributes according to schema.")

        Logger.log(f"end Node __init__(self)")

    def get_id(self):
        """Return node ID or None."""
        return getattr(self, "n_id", None)

    def get_attributes(self):
        """Return attributes as dict."""
        return self.__dict__

    def get_attribute(self, attribute_name):
        """Return attribute value or None."""
        return getattr(self, attribute_name, None)

    @staticmethod
    def safe_cast(value, expected_type):
        """Cast value to expected_type; handle common cases."""
        try:
            if expected_type == bool:
                return str(value).strip().lower() in ["true", "1", "yes"]
            return expected_type(value)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid value: '{value}' is not of type {expected_type.__name__}")

    def set_attribute(self, attribute_name, value):
        """Set attribute value after schema/type check."""
        if self.schema and attribute_name not in self.schema:
            raise ValueError(f"Invalid attribute '{attribute_name}' according to schema.")
        expected_type = self.schema.get(attribute_name, str)
        value = self.safe_cast(value, expected_type)
        setattr(self, attribute_name, value)
        if attribute_name not in self.attributes:
            self.attributes.append(attribute_name)

    def validate_attributes(self):
        """Return True if all set attributes are allowed by schema."""
        Logger.log("Starting attribute validation.")
        Logger.log(f"Schema: {self.schema}")

        for attr in self.attributes:
            Logger.log(f"Validating attribute: {attr}")
            if self.schema and attr not in self.schema:
                Logger.log(f"Attribute '{attr}' not in schema. Validation failed.")
                return False

        Logger.log("All attributes are valid.")
        return True

    @classmethod
    def get_schema(cls):
        """Return schema dict mapping attribute names to types."""
        return cls.schema
