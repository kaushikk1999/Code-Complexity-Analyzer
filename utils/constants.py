"""Application constants."""

APP_NAME = "Complexity Lab"
APP_TAGLINE = "Interview-ready Python complexity, benchmark, and optimization analysis."
DEFAULT_REPEAT_COUNT = 5
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_CODE_CHARS = 20_000
HISTORY_LIMIT = 12

DEFAULT_CODE = '''def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
'''

DEFAULT_INPUT = '{"args": [[2, 7, 11, 15, 21, 30, 42, 55], 57]}'
