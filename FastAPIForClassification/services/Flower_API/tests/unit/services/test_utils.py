import logging
from io import StringIO
import sys
from src.services.utils import setup_logger

def test_setup_logger():
    logging.getLogger().handlers = []
    captured_output = StringIO()
    sys.stdout = captured_output

    logger = setup_logger()

    test_message = "Test logging message"
    logger.info(test_message)

    sys.stdout = sys.__stdout__

    logged_output = captured_output.getvalue()
    assert test_message in logged_output
    assert "INFO" in logged_output

