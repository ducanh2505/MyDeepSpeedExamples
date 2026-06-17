
# Constants
# Training constants
DEFAULT_OPTIMIZER_LR = 0.001
DEFAULT_OPTIMIZER_BETAS = (0.9, 0.999)
BYTES_TO_GB = 1e9
MS_PER_SECOND = 1000
TFLOPS_DENOMINATOR = 1e12


#LOGGER constants
LOGGER_RANK = 0
LOGGER_NAME = "finetune_zero3"
LOGGER_LEVEL = "INFO"
LOGGER_DATE_FMT = '%Y-%m-%d %H:%M:%S'
LOGGER_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# Alpaca dataset formatting
ALPACA_INSTRUCTION_TEMPLATE = "### Instruction:\n{instruction}\n\n"
ALPACA_INPUT_TEMPLATE = "### Input:\n{input}\n\n"
ALPACA_RESPONSE_TEMPLATE = "### Response:\n{output}"