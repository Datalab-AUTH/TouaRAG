"""
This module provides the Config class for loading and managing configuration 
settings from a YAML file.
Classes:
    Config: A class used to load and manage configuration settings from a YAML file.
Usage example:
    config = Config("path/to/config.yaml")
    db_settings = config.get_section("database")
"""
import yaml

class Config:
    """
    A class used to load and manage configuration settings from a YAML file.

    Attributes
    ----------
    config : dict
        A dictionary containing the configuration settings.

    Methods
    -------
    __init__(config_path="config.yaml")
        Initializes the Config object and loads the configuration from the specified YAML file.
    
    get_section(section)
        Retrieves a specific section from the configuration.
    
    load_config(file_path)
        Loads the configuration from the specified YAML file.
    """
    def __init__(self, config_path="./config.yaml"):
        self.config = self.load_config(config_path)

    def get_section(self, section):
        """
        Retrieve a configuration section.

        Args:
            section (str): The name of the section to retrieve from the configuration.

        Returns:
            dict: The configuration settings for the specified section. If the section
                  does not exist, an empty dictionary is returned.
        """
        return self.config.get(section, {})

    def load_config(self, file_path):
        """
        Loads a YAML configuration file.

        Args:
            file_path (str): The path to the YAML configuration file.

        Returns:
            dict: The configuration data loaded from the file.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
