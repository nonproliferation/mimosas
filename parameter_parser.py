import argparse
import configparser
import os.path
import sys

## Parameter parser object
class Parameters:
    def __init__(self, main_path):
        """
        init function
        
        @params:
            main_path   - Required  : absolute path of main.py for reference to other parts of this repository (Str)
        """
        ## Absolute path for main.py -- used to generate paths in the case that main.py isn't called directly from its directory
        self.main_path = main_path
        
        # Default config file path
        default_config_file_path = os.path.join(self.main_path, 'default.config')
        license_file_path = os.path.join(self.main_path, 'LICENSE.txt')
        
        # Parser for parsing config file path options
        parser = argparse.ArgumentParser(description='Configuration file for the analysis process.')
        parser.add_argument('-lic', '--license', dest='license', action='store_true', help='Print software license')
        parser.add_argument('-config', '--config', dest='config_file', help='File path to custom config file')
        args = parser.parse_args()
        
        if (args.license):
            self.print_license(license_file_path)
        
        # If no config file path is passed in as a parameter, load (generate if does not exist) default config file
        if (args.config_file is None):
            ## Config object
            self.config = self.load_config(default_config_file_path)
            
        # If config file path is passed in, load (generate if does not exist) passed in config file
        else:
            self.config = self.load_config(args.config_file)
    
    def print_license(self, license_file):
        """
        Print lines of the license file to console
        
        @params:
            license_file   - Required  : file path to where license file is located (Str)
        """
        # Read lines of the license file and print it out
        with open(license_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            print(line, end =" ")
        print()
    
    def load_config(self, config_file):
        """
        Load config file, or generate a new one if it does not exist already
        
        @params:
            config_file   - Required  : file path to where config file is located (Str)
        """
        # Generate config file if does not exist already
        if (not os.path.exists(config_file)):
            self.generate_config_file(config_file)
        
        ## Config file
        self.config_file = config_file
        config = configparser.ConfigParser()
        config.read(config_file)
        
        if (self.check_config(config)):
            return config
        else:
            sys.exit()
            return None
        
    def check_config(self, config_file):
        """
        Not yet implemented... at the end, once the config file has been finalized, check all values are within operable range
        
        @params:
            config_file   - Required  : file path to where config file is located (Str)
        """
        print('Checking config file', '[NOT IMPLEMENTED]')
        return True
    
    def generate_config_file(self, config_file):
        """
        Copies source.config to config_file, ignoring the first 2 lines (warning to not edit source.config).
        
        @params:
            config_file   - Required  : file path to where new config file should be generated (Str)
        """
        # First open source.config and read in all lines
        with open(os.path.join(self.main_path, 'source.config'), "r") as f:
            lines = f.readlines()
            
        # Write out lines to destination while ignoring the first 2 lines
        with open(config_file, "w") as f:
            for line in lines[2:]:
                f.write(line)
            