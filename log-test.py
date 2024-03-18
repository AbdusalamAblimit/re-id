from loguru import logger
from tqdm import tqdm
import time
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""))
logger.add('log-test.log')
logger.info("Initializing")

for x in tqdm(range(100)):
    logger.info("aaaIterating #{}", x)
    time.sleep(0.1)