import logging
import os
import configparser

_log_initialized = False
logger = None

def init_logger(config_path='config.ini'):
    global logger, _log_initialized
    if _log_initialized:
        return logger
    logger = logging.getLogger('ml_pipeline_logger')
    # Read debug flag from config
    config = configparser.ConfigParser()
    config.read(config_path)
    debug = False
    if 'Logging' in config and 'debug' in config['Logging']:
        debug = config.getboolean('Logging', 'debug')
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    log_file = 'logs/pipeline.log'
    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        c_handler.setLevel(level)
        f_handler.setLevel(level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    _log_initialized = True
    return logger

def get_logger():
    global logger
    if logger is None:
        return init_logger()
    return logger 