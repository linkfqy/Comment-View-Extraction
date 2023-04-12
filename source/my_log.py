import logging
import logging.handlers


def get_logger(filename):

    logger = logging.getLogger("dl-logger")

    handler_console = logging.StreamHandler()
    handler_file = logging.FileHandler(filename=filename, mode='w')

    logger.setLevel(logging.DEBUG)
    handler_console.setLevel(logging.DEBUG)
    handler_file.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(message)s",
                                  datefmt="%y-%m-%d %H:%M:%S")
    handler_console.setFormatter(formatter)
    handler_file.setFormatter(formatter)

    logger.addHandler(handler_console)
    logger.addHandler(handler_file)

    return logger


if __name__ == '__main__':
    logger = get_logger("./save/draft.log")
    logger.debug('This is a customer debug message')
    logger.info('This is an customer info message')
    logger.warning('This is a customer warning message')
    logger.error('This is an customer error message')
    logger.critical('This is a customer critical message')
