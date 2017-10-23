import logging
import logging.handlers
try:
    reload(logging) # reload the module to avoid multiple Spyder console output
except:
    pass


def start_logging(log_level='info', logfile=None):
    """
    Use this to start logging to console.
    You might want to specify log file also (filename.log)
    """
    if log_level is None:
        main_logger = logging.getLogger()
        return main_logger
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    hndlrs = [logging.StreamHandler()] # console handler
    if logfile is not None:
        hndlrs.append(logging.FileHandler('{}'.format(logfile),
                                                   encoding='utf8'))
    main_logger = logging.getLogger()
    main_logger.setLevel(LEVELS[log_level])
    #create formatter
    formatter = logging.Formatter("%(message)s (%(levelname)s:%(name)s at %(asctime)s)")

    for h in hndlrs:
        h.setFormatter(formatter)
        h.setLevel(LEVELS[log_level])
        main_logger.addHandler(h)
    main_logger.info("\nLogging started")
    return main_logger
