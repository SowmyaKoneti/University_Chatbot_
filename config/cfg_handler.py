

import os
from configparser import ConfigParser


class CfgHandlerError(Exception):
    pass


class CfgHandler(ConfigParser, object):
    

    _defaultConfigFileName = "cfg.ini"
    _configFileInUse = None

    def __init__(self, cfg_file=None):
       
        super(CfgHandler, self).__init__()
        self.load_configuration(cfg_file)

    def load_configuration(self, cfg_file):
        
        self._configFileInUse = (os.path.abspath(os.path.dirname(__file__)) + os.sep +
                                 self._defaultConfigFileName) if cfg_file is None else cfg_file

        lst = self.read(self._configFileInUse)

        if len(lst) == 0:
            raise CfgHandlerError("Config file not found")

    def get_cfg_file_in_use(self):
        return self._configFileInUse
