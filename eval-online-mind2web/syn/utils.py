import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, format='<green>{time:YY-MM-DD HH:mm:ss.SS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>\n<level>{message}</level>', level='DEBUG')
